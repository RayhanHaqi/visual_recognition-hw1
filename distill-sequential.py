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

# ==========================================
# 1. KONFIGURASI UTAMA
# ==========================================
# PATH GURU (Ganti dengan path checkpoint 0.95 kamu)
TEACHER_PATH = '/home/tilakoid/selectedtopics/cv_hw1_data/checkpoints/teacher_vit_so400m_patch14_siglip_378.webli_ft_in1k_run2_best.pth'
TEACHER_MODEL_NAME = 'vit_so400m_patch14_siglip_378.webli_ft_in1k'

# CONFIG MURID
STUDENT_MODEL_NAME = 'resnetrs200.tf_in1k'
DATA_DIR = '/home/tilakoid/selectedtopics/cv_hw1_data/data'
NUM_CLASSES = 100
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DYNAMIC BATCH & ACCUMULATION (Target 64)
PHYSICAL_BATCH_SIZE = 32 # Sesuai request kamu
TARGET_BATCH = 64
ACC_STEPS = max(1, TARGET_BATCH // PHYSICAL_BATCH_SIZE)

# DISTILLATION PARAMS
TEMPERATURE = 3.0
ALPHA = 0.7 # Lebih percaya Guru (0.95+)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ==========================================
# 2. CUSTOM DATASET FOR DISTILLATION
# ==========================================
class DistillDataset(Dataset):
    def __init__(self, base_dataset, soft_labels):
        self.base_dataset = base_dataset
        self.soft_labels = soft_labels # Logits dari Guru

    def __getitem__(self, index):
        img, hard_label = self.base_dataset[index]
        soft_label = self.soft_labels[index]
        return img, hard_label, soft_label

    def __len__(self):
        return len(self.base_dataset)

# ==========================================
# 3. FASE 1: LABELING (GURU BEKERJA)
# ==========================================
def run_labeling():
    print(f"\n🧠 FASE 1: Guru ({TEACHER_MODEL_NAME}) melabeli dataset...")
    
    # Setup Resolusi Guru (SigLIP 448)
    img_size = 448 if "siglip" in TEACHER_MODEL_NAME else 320
    
    # Transform Statis (Hanya resize, tanpa acak agar label konsisten)
    transform_labeling = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Sesuaikan SigLIP
    ])

    base_train = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform_labeling)
    loader = DataLoader(base_train, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, num_workers=4)

    # Load Teacher
    teacher = timm.create_model(TEACHER_MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, img_size=img_size)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE))
    teacher.to(DEVICE).eval()

    all_soft_labels = torch.zeros((len(base_train), NUM_CLASSES), dtype=torch.float16)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(loader, desc="Labeling")):
            with torch.amp.autocast('cuda'):
                logits = teacher(inputs.to(DEVICE))
                # Simpan logits di CPU dalam float16 agar hemat RAM
                all_soft_labels[i*PHYSICAL_BATCH_SIZE : (i+1)*PHYSICAL_BATCH_SIZE] = logits.cpu().half()

    # BERSIHKAN GPU UNTUK MURID
    del teacher
    torch.cuda.empty_cache()
    gc.collect()
    print("✅ Labeling selesai. GPU dibersihkan.")
    return all_soft_labels

# ==========================================
# 4. FASE 2: TRAINING (MURID BELAJAR)
# ==========================================
def train_distill(soft_labels):
    print(f"\n🔥 FASE 2: Training Murid ({STUDENT_MODEL_NAME})...")
    start_time = time.time()
    
    # Resolusi Murid (ResNet-RS 320px)
    img_size = 320
    
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

    # Dataset & Loader
    base_train = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform_train)
    distill_train_ds = DistillDataset(base_train, soft_labels)
    
    train_loader = DataLoader(distill_train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform_val), 
                            batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, num_workers=4)

    # Model Murid
    model = timm.create_model(STUDENT_MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES, drop_path_rate=0.1).to(DEVICE)
    model_ema = ModelEmaV2(model, decay=0.9998)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=4e-4, steps_per_epoch=len(train_loader)//ACC_STEPS, epochs=EPOCHS, pct_start=0.15
    )
    
    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0.0

    # Training Loop
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
                
                # KD LOSS LOGIC
                # KL Divergence untuk Soft Labels (Ilmu Guru)
                loss_soft = F.kl_div(
                    F.log_softmax(student_logits / TEMPERATURE, dim=1),
                    F.softmax(soft_labels_batch / TEMPERATURE, dim=1),
                    reduction='batchmean'
                ) * (TEMPERATURE ** 2)
                
                # Cross Entropy untuk Hard Labels (Kebenaran Dataset)
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

        # VALIDASI
        model_ema.module.eval()
        top1_correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    out = model_ema.module(inputs)
                top1_correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
        
        val_acc = 100. * top1_correct / total
        print(f"📊 Ep {epoch} | Loss: {train_loss_total/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_ema.module.state_dict(), f"checkpoints/distill_{STUDENT_MODEL_NAME}_best.pth")

if __name__ == "__main__":
    # Step 1: Dapatkan Label dari Guru
    computed_soft_labels = run_labeling()
    
    # Step 2: Latih Murid
    train_distill(computed_soft_labels)