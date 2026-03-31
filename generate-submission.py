import torch
import timm
import os
import pandas as pd
import zipfile
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ==========================================
# 1. KONFIGURASI (Cukup Masukkan Path Checkpoint)
# ==========================================
# Masukkan path lengkap ke file .pth kamu di sini
CHECKPOINT_PATH = '/home/tilakoid/selectedtopics/cv_hw1_data/checkpoints/distill_resnetrs200.tf_in1k_run1_best.pth'

DATA_DIR = '/home/tilakoid/selectedtopics/cv_hw1_data/data/' 
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
SUBMISSION_DIR = 'submissions'

os.makedirs(SUBMISSION_DIR, exist_ok=True)

# --- LOGIKA EKSTRAKSI NAMA MODEL OTOMATIS ---
filename = os.path.basename(CHECKPOINT_PATH)
base_output_name = filename.replace('.pth', '')

# Membersihkan prefix (teacher/distill) dan suffix (run/best/mode) untuk mendapatkan string arsitektur timm
temp_name = filename.replace('teacher_', '').replace('distill_', '')
for suffix in ['_best', '_run', '_fine_tune', '_scratch', '.pth']:
    if suffix in temp_name:
        temp_name = temp_name.split(suffix)[0]

MODEL_NAME = 'resnetrs200.tf_in1k'
print(f"🔎 Terdeteksi Arsitektur: {MODEL_NAME}")

OUTPUT_CSV = os.path.join(SUBMISSION_DIR, f"{base_output_name}.csv")
OUTPUT_ZIP = os.path.join(SUBMISSION_DIR, f"{base_output_name}.zip")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

# ==========================================
# 2. LOAD MODEL & MAPPING
# ==========================================
# Deteksi resolusi otomatis agar hasil prediksi akurat (sesuai setting training)
img_size = 448 # Default
# if "siglip" in MODEL_NAME or "eva" in MODEL_NAME:
#     img_size = 448
# elif "convnext" in MODEL_NAME or "resnetrs" in MODEL_NAME or "resnest" in MODEL_NAME:
#     img_size = 320

print(f"📏 Menggunakan Resolusi: {img_size}px")

# Inisialisasi model dan load bobot
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=100)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Mapping folder ke label (PENTING: Urutan alfabetis sesuai ImageFolder)
alphabetical_folders = sorted(os.listdir(TRAIN_DIR))

# Mengambil konfigurasi normalisasi asli dari model
data_config = timm.data.resolve_data_config({}, model=model)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
])

# ==========================================
# 3. INFERENCE
# ==========================================
results = []
test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"🚀 Memproses {len(test_files)} gambar test...")
with torch.no_grad():
    for filename in tqdm(test_files):
        img_path = os.path.join(TEST_DIR, filename)
        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                output = model(image)
            
            pred_idx = output.argmax(dim=1).item()
            
            # Konversi index ke nama folder, lalu ke integer label
            actual_folder_name = alphabetical_folders[pred_idx]
            correct_label = int(actual_folder_name) 

            image_id = os.path.splitext(filename)[0] 
            results.append({'image_name': image_id, 'pred_label': correct_label})
        except Exception as e:
            print(f"Gagal memproses {filename}: {e}")

# ==========================================
# 4. SIMPAN HASIL
# ==========================================
df = pd.DataFrame(results)
# Sort berdasarkan nama gambar agar rapi (opsional)
df = df.sort_values('image_name')
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ CSV tersimpan: {OUTPUT_CSV}")

with zipfile.ZipFile(OUTPUT_ZIP, 'w') as z:
    z.write(OUTPUT_CSV, arcname='prediction.csv')
print(f"📦 ZIP siap submit: {OUTPUT_ZIP}")