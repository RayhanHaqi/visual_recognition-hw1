import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from timm.utils import ModelEmaV2
import os
import time
import csv
import gc
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

# ==========================================
# 0. SISTEM PELACAKAN OTOMATIS (RUN ID)
# ==========================================
TRACKER_FILE = "run_tracker_student.txt"
if os.path.exists(TRACKER_FILE):
    with open(TRACKER_FILE, "r") as f:
        RUN_ID = int(f.read().strip()) + 1
else:
    RUN_ID = 2 

with open(TRACKER_FILE, "w") as f: f.write(str(RUN_ID))
print(f"🔥 MEMULAI STUDENT RUN {RUN_ID}: FULL METRICS MODE")

# ==========================================
# 1. KONFIGURASI STUDENT
# ==========================================
MODELS = ['resnetrs200.tf_in1k', 'resnest200e.in1k']
DATA_DIR = '/home/tilakoid/selectedtopics/cv_hw1_data/data'
NUM_CLASSES = 100
EPOCHS = 30      
IMG_SIZE = 320   # Menggunakan resolusi tinggi hasil eksperimen sebelumnya
INITIAL_BATCH_SIZE = 32 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)
SUMMARY_CSV = "results/student_summary.csv"

def save_summary(data):
    file_exists = os.path.isfile(SUMMARY_CSV)
    with open(SUMMARY_CSV, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(data)

# ==========================================
# 2. TRAINING ENGINE
# ==========================================
def train_student(model_name, batch_size):
    start_time = time.time()
    print(f"\n🚀 Training Student: {model_name} @ {IMG_SIZE}px")
    
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES, drop_path_rate=0.1).to(DEVICE)
    model_ema = ModelEmaV2(model, decay=0.9998)

    # Transform
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
        optimizer, max_lr=4e-4, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.15
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    best_metrics = {}

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0.0
        pbar = tqdm(train_loader, desc=f"Run {RUN_ID} | Ep {epoch}/{EPOCHS}")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            model_ema.update(model)
            scheduler.step()
            train_loss_total += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- VALIDASI LENGKAP (Top-1, Top-5, TTA, Precision, Recall, F1) ---
        model_ema.module.eval()
        top1_correct, top5_correct, total = 0, 0, 0
        all_preds, all_targets = [], []
        val_loss_total = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    # TTA: Average Original + Flip
                    out = (model_ema.module(inputs) + model_ema.module(torch.flip(inputs, [3]))) / 2.0
                    loss_v = criterion(out, labels)
                
                val_loss_total += loss_v.item()
                
                # Akurasi Top-1
                _, pred1 = out.topk(1, 1, True, True)
                top1_correct += (pred1.flatten() == labels).sum().item()
                
                # Akurasi Top-5
                _, pred5 = out.topk(5, 1, True, True)
                top5_correct += (pred5 == labels.view(-1, 1)).sum().item()
                
                total += labels.size(0)
                all_preds.extend(pred1.flatten().cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        # Kalkulasi Metrik Final
        val_acc = 100. * top1_correct / total
        top5_acc = 100. * top5_correct / total
        avg_val_loss = val_loss_total / len(val_loader)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )
        
        log_msg = f"Epoch {epoch} | T-Loss: {train_loss_total/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Top-5: {top5_acc:.2f}% | F1: {f1*100:.2f}%"
        print(f"📊 {log_msg}")
        with open(f"logs/{model_name}_run{RUN_ID}.txt", "a") as f: f.write(f"{log_msg}\n")
            
        if val_acc > best_acc:
            best_acc = val_acc
            best_metrics = {
                "Run_ID": RUN_ID,
                "Model": model_name,
                "Epoch": epoch,
                "Best_Top1_Acc": f"{val_acc:.2f}",
                "Top5_Acc": f"{top5_acc:.2f}",
                "F1_Score": f"{f1*100:.2f}",
                "Precision": f"{precision*100:.2f}",
                "Recall": f"{recall*100:.2f}",
                "Val_Loss": f"{avg_val_loss:.4f}",
                "Final_BS": batch_size,
                "Time_Min": f"{(time.time() - start_time) / 60:.1f}"
            }
            torch.save(model_ema.module.state_dict(), f"checkpoints/{model_name}_best_run{RUN_ID}.pth")

    return best_metrics

if __name__ == "__main__":
    for model_name in MODELS:
        current_bs = INITIAL_BATCH_SIZE
        success = False
        while not success and current_bs >= 8:
            try:
                m = train_student(model_name, current_bs)
                save_summary(m)
                success = True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    current_bs -= 4
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"⚠️ OOM. Menurunkan batch size ke {current_bs}...")
                else: raise e