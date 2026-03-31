import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from timm.utils import ModelEmaV2
from timm.data.mixup import Mixup
import os
import time
from tqdm import tqdm

# ==========================================
# 0. TRACKER & PATHS (PASTIKAN PATH BENAR)
# ==========================================
TRACKER_FILE = "run_tracker_multi_distill.txt"
if os.path.exists(TRACKER_FILE):
    with open(TRACKER_FILE, "r") as f: RUN_ID = int(f.read().strip()) + 1
else: RUN_ID = 1
with open(TRACKER_FILE, "w") as f: f.write(str(RUN_ID))

# Daftar Guru dan Path Checkpoint-nya
TEACHERS_CONFIG = {
    'siglip': {
        'name': 'vit_so400m_patch14_siglip_378.webli_ft_in1k',
        'path': 'checkpoints/teacher_vit_so400m_patch14_siglip_378.webli_ft_in1k_run2_best.pth'
    },
    'convnext': {
        'name': 'convnext_xxlarge.clip_laion2b_soup_ft_in1k',
        'path': 'checkpoints/teacher_convnext_xxlarge.clip_laion2b_soup_ft_in1k_run2_best.pth'
    },
    'eva': {
        'name': 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
        'path': 'checkpoints/teacher_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k_run2_best.pth'
    }
}

STUDENT_NAME = 'resnetrs200.tf_in1k'
DATA_DIR = '/home/tilakoid/selectedtopics/cv_hw1_data/data'
NUM_CLASSES = 100
EPOCHS = 20
IMG_SIZE = 320 # Mengikuti Run 3 Student
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
ALPHA = 0.4        # Bobot Label Asli (0.4) vs Guru (0.6)
TEMPERATURE = 4.0
BATCH_SIZE = 4     # Kecil karena memuat 4 model sekaligus
ACC_STEPS = 8      # Efektif Batch Size = 32

# ==========================================
# 1. SETUP MODELS
# ==========================================
def load_all_models():
    teachers = {}
    for key, cfg in TEACHERS_CONFIG.items():
        print(f"🧠 Loading Teacher {key.upper()}...")
        m = timm.create_model(cfg['name'], pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
        m.load_state_dict(torch.load(cfg['path']))
        m.eval()
        for p in m.parameters(): p.requires_grad = False
        teachers[key] = m

    print(f"🎓 Loading Student: {STUDENT_NAME}")
    student = timm.create_model(STUDENT_NAME, pretrained=True, num_classes=NUM_CLASSES, drop_path_rate=0.1).to(DEVICE)
    return teachers, student

# ==========================================
# 2. MULTI-TEACHER LOSS
# ==========================================
def multi_distill_loss(student_logits, teachers_logits_list, labels, T, alpha):
    # Loss 1: Hard Label (Label Asli)
    loss_ce = F.cross_entropy(student_logits, labels)
    
    # Loss 2: Soft Labels (Rata-rata KLD dari semua Guru)
    loss_kld = 0
    for t_logits in teachers_logits_list:
        loss_kld += F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(t_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)
    
    loss_kld = loss_kld / len(teachers_logits_list)
    return alpha * loss_ce + (1. - alpha) * loss_kld

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def run_multi_distill():
    teachers, student = load_all_models()
    student_ema = ModelEmaV2(student, decay=0.9999)
    
    # Mixup Config
    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch', num_classes=NUM_CLASSES)

    # Transform (Gunakan mean/std dari model terbanyak, biasanya ImageNet standar)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), 
                            transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])), 
                            batch_size=BATCH_SIZE, shuffle=False)
    
    train_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform_train), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(student.parameters(), lr=5e-5, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader)//ACC_STEPS, epochs=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    for epoch in range(EPOCHS):
        student.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Multi-Distill Run {RUN_ID} | Ep {epoch}")
        
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Forward Teachers (No Grad)
            teachers_logits = []
            with torch.no_grad():
                for t in teachers.values():
                    teachers_logits.append(t(inputs))
            
            # Forward Student
            with torch.amp.autocast('cuda'):
                student_logits = student(inputs)
                loss = multi_distill_loss(student_logits, teachers_logits, labels, TEMPERATURE, ALPHA) / ACC_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACC_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                student_ema.update(student)
                scheduler.step()
            
            pbar.set_postfix({'loss': f"{loss.item()*ACC_STEPS:.4f}"})

        # Eval with TTA
        student_ema.module.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                out = (student_ema.module(inputs) + student_ema.module(torch.flip(inputs, [3]))) / 2.0
                correct += (out.argmax(1) == labels.to(DEVICE)).sum().item()
                total += labels.size(0)
        
        val_acc = 100. * correct / total
        print(f"📊 Run {RUN_ID} | Multi-Distill Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student_ema.module.state_dict(), f"checkpoints/multi_distill_{STUDENT_NAME}_run{RUN_ID}_best.pth")

if __name__ == "__main__":
    run_multi_distill()