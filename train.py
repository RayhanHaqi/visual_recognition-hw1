import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
import time
from tqdm import tqdm

# ==========================================
# 1. KONFIGURASI & LOGGING
# ==========================================
MODEL_NAME = 'resnetrs200.tf_in1k' 
MODE = 'fine_tune' # Pilih: 'fine_tune', 'scratch', 'frozen_back'
DATA_DIR = './data'
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"{MODEL_NAME}_{MODE}_amp_log.txt")

def log_print(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

# ==========================================
# 2. MODEL & DATA SETUP
# ==========================================
# Load model dengan timm
is_pretrained = False if MODE == 'scratch' else True
model = timm.create_model(MODEL_NAME, pretrained=is_pretrained, num_classes=100)

if MODE == 'frozen_back':
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, 'get_classifier'):
        for param in model.get_classifier().parameters():
            param.requires_grad = True

model = model.to(DEVICE)

# Data Augmentation (Standard SOTA)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform), 
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform), 
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# --- INITIALIZE AMP SCALER ---
scaler = torch.cuda.amp.GradScaler()

# ==========================================
# 3. TRAINING LOOP DENGAN AMP
# ==========================================
log_print(f"🚀 Memulai Eksperimen: {MODEL_NAME} ({MODE}) dengan AMP")

best_acc = 0.0
for epoch in range(EPOCHS):
    start_time = time.time()
    
    # --- PHASE: TRAIN ---
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # forward pass dengan autocast
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # backward pass dengan scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f"{loss.item():.3f}", 'acc': f"{100.*train_correct/train_total:.2f}%"})

    # --- PHASE: VAL ---
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * val_correct / val_total
    epoch_time = time.time() - start_time
    
    # Simpan jika akurasi terbaik
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/{MODEL_NAME}_{MODE}_best.pth")
        save_msg = " (Saved Best!)"
    else:
        save_msg = ""

    log_print(f"Epoch {epoch} | Jam: {time.strftime('%H:%M:%S')} | Val Acc: {val_acc:.2f}% | Durasi: {epoch_time:.1f}s{save_msg}")

log_print(f"\n✅ Eksperimen Selesai! Best Val Acc: {best_acc:.2f}%")