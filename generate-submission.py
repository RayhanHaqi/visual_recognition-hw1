import torch
import timm
import os
import pandas as pd
import zipfile
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

CHECKPOINT_PATH = '/home/tilakoid/selectedtopics/cv_hw1_data/checkpoints/distill_resnetrs200.tf_in1k-vit_so400m_patch14_siglip_378.pth'

DATA_DIR = '/home/tilakoid/selectedtopics/cv_hw1_data/data/' 
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
SUBMISSION_DIR = 'submissions'

os.makedirs(SUBMISSION_DIR, exist_ok=True)

filename = os.path.basename(CHECKPOINT_PATH)
base_output_name = filename.replace('.pth', '')

temp_name = filename.replace('teacher_', '').replace('distill_', '')
for suffix in ['_best', '_run', '_fine_tune', '_scratch', '.pth']:
    if suffix in temp_name:
        temp_name = temp_name.split(suffix)[0]

MODEL_NAME = 'resnetrs200.tf_in1k'

OUTPUT_CSV = os.path.join(SUBMISSION_DIR, f"{base_output_name}.csv")
OUTPUT_ZIP = os.path.join(SUBMISSION_DIR, f"{base_output_name}.zip")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_size = 448


model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=100)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

alphabetical_folders = sorted(os.listdir(TRAIN_DIR))

data_config = timm.data.resolve_data_config({}, model=model)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
])

results = []
test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

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
            print(f"Failed to process {filename}: {e}")


df = pd.DataFrame(results)
df = df.sort_values('image_name')
df.to_csv(OUTPUT_CSV, index=False)

with zipfile.ZipFile(OUTPUT_ZIP, 'w') as z:
    z.write(OUTPUT_CSV, arcname='prediction.csv')
print(f"📦 ZIP ready to submit: {OUTPUT_ZIP}")