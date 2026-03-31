import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from timm.utils import ModelEmaV2
from timm.data.mixup import Mixup
import os
import time
import csv
import gc
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


TRACKER_FILE = "run_tracker_student.txt"
if os.path.exists(TRACKER_FILE):
    with open(TRACKER_FILE, "r") as f:
        RUN_ID = int(f.read().strip()) + 1
else:
    RUN_ID = 10 

with open(TRACKER_FILE, "w") as f: f.write(str(RUN_ID))
print(f"INSANE STUDENT RUN {RUN_ID} | TARGET >0.96 | RESOLUTION: 448px | EPOCHS: 75")


MODELS = ['resnetrs200.tf_in1k']
DATA_DIR = '/home/tilakoid/selectedtopics/cv_hw1_data/data'
NUM_CLASSES = 100
EPOCHS = 75
IMG_SIZE = 448 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PHYSICAL_BATCH_SIZE = 4  
ACC_STEPS = 16          

mixup_fn = Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, 
    switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=NUM_CLASSES
)

os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)
SUMMARY_CSV = "results/student_insane_summary.csv"

def save_summary(data):
    file_exists = os.path.isfile(SUMMARY_CSV)
    with open(SUMMARY_CSV, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(data)

def log_epoch_progress(model_name, run_id, epoch_data):
    filename = f"logs/insane_{model_name}_run{run_id}_progress.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=epoch_data.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(epoch_data)

def train_student(model_name, batch_size):
    start_time = time.time()
    print(f"\n🚀 Processing Student (Insane Mode): {model_name}")
    
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES, 
                             drop_path_rate=0.15).to(DEVICE) 
    
    model_ema = ModelEmaV2(model, decay=0.9999) 

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform_train), 
                              batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform_val), 
                            batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader)//ACC_STEPS, epochs=EPOCHS, pct_start=0.1
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    best_metrics = {}

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        train_loss_total = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS}")
        
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            inputs, labels_mixed = mixup_fn(inputs, labels)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels_mixed) / ACC_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACC_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                model_ema.update(model)
                scheduler.step()
            
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
                    loss_v = criterion(out, labels)
                
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
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )
        
        epoch_results = {
            "Epoch": epoch, "T_Loss": f"{avg_train_loss:.4f}", "V_Loss": f"{avg_val_loss:.4f}",
            "Top1_Acc": f"{val_acc:.2f}", "Top5_Acc": f"{top5_acc:.2f}", 
            "F1_Score": f"{f1*100:.2f}", "Precision": f"{precision*100:.2f}", "Recall": f"{recall*100:.2f}"
        }
        log_epoch_progress(model_name, RUN_ID, epoch_results)

        if val_acc > best_acc:
            best_acc = val_acc
            best_metrics = {
                "Run_ID": RUN_ID, "Model": model_name, "Epoch": epoch, "Best_Top1_Acc": f"{val_acc:.2f}",
                "Top5_Acc": f"{top5_acc:.2f}", "F1_Score": f"{f1*100:.2f}", "Precision": f"{precision*100:.2f}",
                "Recall": f"{recall*100:.2f}", "Val_Loss": f"{avg_val_loss:.4f}", "Final_BS": batch_size,
                "Time_Min": f"{(time.time() - start_time) / 60:.1f}"
            }
            torch.save(model_ema.module.state_dict(), f"checkpoints/insane_{model_name}_run{RUN_ID}_best.pth")

    if best_metrics: save_summary(best_metrics)

if __name__ == "__main__":
    for model_name in MODELS:
        try:
            train_student(model_name, PHYSICAL_BATCH_SIZE)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
            else: raise e