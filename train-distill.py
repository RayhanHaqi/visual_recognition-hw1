import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import timm
from timm.utils import ModelEmaV2
from timm.data.mixup import Mixup
import os
import time
import csv
import gc
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

TRACKER_FILE = "run_tracker_distill.txt"
if os.path.exists(TRACKER_FILE):
    with open(TRACKER_FILE, "r") as f:
        RUN_ID = int(f.read().strip()) + 1
else:
    RUN_ID = 1 

with open(TRACKER_FILE, "w") as f: f.write(str(RUN_ID))
print(f"🔥 DISTILLATION RUN {RUN_ID} | TARGET: 0.95+")

TEACHER_PATH = '/home/tilakoid/selectedtopics/cv_hw1_data/checkpoints/vit_so400m_patch14_siglip_378.webli_ft_in1k_teacher_best.pth'
TEACHER_MODEL_NAME = 'vit_so400m_patch14_siglip_378.webli_ft_in1k'

STUDENT_MODEL_NAME = 'resnetrs200.tf_in1k'
STUDENT_PATH = '/home/tilakoid/selectedtopics/cv_hw1_data/checkpoints/insane_resnetrs200.tf_in1k_run8_best.pth' 

DATA_DIR = '/home/tilakoid/selectedtopics/cv_hw1_data/data'
NUM_CLASSES = 100
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TEACHER_BATCH_SIZE = 8  
PHYSICAL_BATCH_SIZE = 16 
TARGET_BATCH = 64
ACC_STEPS = max(1, TARGET_BATCH // PHYSICAL_BATCH_SIZE)

TEMPERATURE = 4.0
ALPHA = 0.9 
SOFT_LABELS_PATH = f'checkpoints/teacher_soft_labels_run{RUN_ID}.pt'

os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)
SUMMARY_CSV = "results/distill_summary.csv"


def save_summary(data):
    file_exists = os.path.isfile(SUMMARY_CSV)
    with open(SUMMARY_CSV, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(data)

def log_epoch_progress(run_id, epoch_data):
    filename = f"logs/distill_run{run_id}_progress.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=epoch_data.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(epoch_data)

class DistillDataset(Dataset):
    def __init__(self, base_dataset, soft_labels):
        self.base_dataset = base_dataset
        self.soft_labels = soft_labels

    def __getitem__(self, index):
        img, hard_label = self.base_dataset[index]
        soft_label = self.soft_labels[index]
        return img, hard_label, soft_label

    def __len__(self):
        return len(self.base_dataset)

def run_labeling():
    print(f"\nPhase 1: Teacher ({TEACHER_MODEL_NAME}) labeling dataset...")
    img_size = 378 if "siglip" in TEACHER_MODEL_NAME else 320
    
    transform_labeling = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])

    base_train = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform_labeling)
    loader = DataLoader(base_train, batch_size=TEACHER_BATCH_SIZE, shuffle=False, num_workers=4)

    teacher = timm.create_model(TEACHER_MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, img_size=img_size)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE))
    teacher.to(DEVICE).eval()

    all_soft_labels = torch.zeros((len(base_train), NUM_CLASSES), dtype=torch.float16)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(loader, desc="Labeling")):
            with torch.amp.autocast('cuda'):
                logits = teacher(inputs.to(DEVICE))
                start_idx = i * TEACHER_BATCH_SIZE
                end_idx = start_idx + inputs.size(0)
                all_soft_labels[start_idx : end_idx] = logits.cpu().half()

    del teacher
    torch.cuda.empty_cache()
    gc.collect()
    
    torch.save(all_soft_labels, SOFT_LABELS_PATH)
    print(f"Soft labels saved to {SOFT_LABELS_PATH}")
    print("Finished labeling. GPU released.")
    return all_soft_labels

def train_distill(soft_labels):
    print(f"\nPhase 2: Training Student ({STUDENT_MODEL_NAME}) from checkpoint...")
    start_time = time.time()
    img_size = 320
    
    model = timm.create_model(STUDENT_MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, drop_path_rate=0.1)
    print(f"Loading student checkpoint: {STUDENT_PATH}")
    model.load_state_dict(torch.load(STUDENT_PATH, map_location='cpu'))
    model.to(DEVICE)
    
    model_ema = ModelEmaV2(model, decay=0.9998)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    base_train = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform_train)
    distill_train_ds = DistillDataset(base_train, soft_labels)
    
    train_loader = DataLoader(distill_train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform_val), 
                            batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, num_workers=4)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader)//ACC_STEPS, epochs=EPOCHS, pct_start=0.15
    )
    
    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0.0
    best_metrics = {}

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        train_loss_total = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS}")

        for i, (inputs, hard_labels, soft_labels_batch) in enumerate(pbar):
            inputs, hard_labels = inputs.to(DEVICE), hard_labels.to(DEVICE)
            soft_labels_batch = soft_labels_batch.to(DEVICE).float()
            
            with torch.amp.autocast('cuda'):
                student_logits = model(inputs)
                loss_soft = F.kl_div(
                    F.log_softmax(student_logits / TEMPERATURE, dim=1),
                    F.softmax(soft_labels_batch / TEMPERATURE, dim=1),
                    reduction='batchmean'
                ) * (TEMPERATURE ** 2)
                loss_hard = F.cross_entropy(student_logits, hard_labels)
                loss = (ALPHA * loss_soft + (1 - ALPHA) * loss_hard) / ACC_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACC_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                model_ema.update(model)

            train_loss_total += loss.item() * ACC_STEPS
            pbar.set_postfix({'loss': f"{loss.item()*ACC_STEPS:.4f}"})

        model_ema.module.eval()
        top1_correct, top5_correct, total = 0, 0, 0
        all_preds, all_targets = [], []
        val_loss_total = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    out = (model_ema.module(inputs) + model_ema.module(torch.flip(inputs, [3]))) / 2.0
                    loss_v = F.cross_entropy(out, labels)
                
                val_loss_total += loss_v.item()
                _, pred1 = out.topk(1, 1, True, True)
                _, pred5 = out.topk(5, 1, True, True)
                
                top1_correct += (pred1.flatten() == labels).sum().item()
                top5_correct += (pred5 == labels.view(-1, 1)).sum().item()
                total += labels.size(0)
                
                all_preds.extend(pred1.flatten().cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        val_acc = 100. * top1_correct / total
        top5_acc = 100. * top5_correct / total
        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
        
        epoch_results = {
            "Epoch": epoch, "T_Loss": f"{avg_train_loss:.4f}", "V_Loss": f"{avg_val_loss:.4f}",
            "Top1_Acc": f"{val_acc:.2f}", "Top5_Acc": f"{top5_acc:.2f}", 
            "F1_Score": f"{f1*100:.2f}", "Precision": f"{precision*100:.2f}", "Recall": f"{recall*100:.2f}"
        }
        log_epoch_progress(RUN_ID, epoch_results)
        print(f"Ep {epoch} | T-Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}% | Top5: {top5_acc:.2f}% | F1: {f1*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_metrics = {
                "Run_ID": RUN_ID, "Student": STUDENT_MODEL_NAME, "Teacher": TEACHER_MODEL_NAME,
                "Epoch": epoch, "Best_Top1_Acc": f"{val_acc:.2f}",
                "Top5_Acc": f"{top5_acc:.2f}", "F1_Score": f"{f1*100:.2f}", "Precision": f"{precision*100:.2f}",
                "Recall": f"{recall*100:.2f}", "Val_Loss": f"{avg_val_loss:.4f}",
                "Time_Min": f"{(time.time() - start_time) / 60:.1f}"
            }
            torch.save(model_ema.module.state_dict(), f"checkpoints/distill_{STUDENT_MODEL_NAME}_run{RUN_ID}_best.pth")

    if best_metrics: 
        save_summary(best_metrics)
        print(f"🏆 Best Metrics Saved: Top-1 Acc {best_acc:.2f}%")

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if os.path.exists(SOFT_LABELS_PATH):
        print(f"Found soft labels at {SOFT_LABELS_PATH}.")
        print("Skipping Phase 1 (Labeling Guru) and loading from RAM...")
        computed_soft_labels = torch.load(SOFT_LABELS_PATH)
    else:
        print("Soft labels not found. Starting Phase 1 from scratch...")
        computed_soft_labels = run_labeling()
    
    train_distill(computed_soft_labels)