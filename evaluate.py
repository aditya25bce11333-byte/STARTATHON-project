import torch
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import FalconOffRoadDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 11

# --- CONFIGURATION ---
TEST_IMG_PATH = r"C:\PROJECT5\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Color_Images"
TEST_MASK_PATH = r"C:\PROJECT5\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Segmentation"
MODEL_WEIGHTS = "falcon_donkey_model.pth"

# 1. Define Evaluation Transforms
eval_transform = A.Compose([
    A.Resize(544, 960), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 2. Load Model
model = smp.DeepLabV3Plus(encoder_name="resnet34", classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
model.eval()

# 3. Load Dataset
test_dataset = FalconOffRoadDataset(TEST_IMG_PATH, TEST_MASK_PATH, transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def calculate_iou(preds, labels, num_classes):
    ious = []
    preds = torch.argmax(preds, dim=1)
    for cls in range(num_classes):
        inter = ((preds == cls) & (labels == cls)).sum().item()
        union = ((preds == cls) | (labels == cls)).sum().item()
        if union == 0:
            ious.append(float('nan')) 
        else:
            ious.append(inter / union)
    return ious

all_ious = []
print(f"Starting Evaluation on {DEVICE}...")

with torch.no_grad():
    for images, masks in tqdm(test_loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        ious = calculate_iou(outputs, masks, NUM_CLASSES)
        all_ious.append(ious)

# --- FIXED PRINTING LOGIC ---
# Convert to a proper 2D numpy array
all_ious_np = np.array(all_ious)

# Calculate Mean across images (axis 0) while ignoring NaNs
# This gives the average IoU for each of the 11 classes
mean_ious = np.nanmean(all_ious_np, axis=0)

print("\n" + "="*40)
print(f"{'CLASS':<15} | {'IoU SCORE':<10} | {'PERCENTAGE'}")
print("-" * 40)

classes = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", 
    "Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

valid_scores = []
for i, score in enumerate(mean_ious):
    class_name = classes[i]
    if np.isnan(score):
        print(f"{class_name:<15} | {'N/A':<10} | (Not in Test Set)")
    else:
        print(f"{class_name:<15} | {score:.4f}     | {score*100:.2f}%")
        valid_scores.append(score)

print("="*40)
if valid_scores:
    total_miou = np.mean(valid_scores)
    print(f"{'OVERALL mIoU':<15} | {total_miou:.4f}     | {total_miou*100:.2f}%")
else:
    print("No valid classes were found to calculate mIoU.")
print("="*40)