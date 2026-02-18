from dataset import FalconOffRoadDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 
NUM_CLASSES = 11
EPOCHS = 15
LEARNING_RATE = 1e-4

train_transform = A.Compose([
    A.Resize(544, 960),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.RandomShadow(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class RobustLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torch.tensor([1.0, 1.5, 3.0, 2.0, 2.5, 5.0, 10.0, 5.0, 2.5, 1.0, 0.1]).to(DEVICE)
        self.dice = smp.losses.DiceLoss(mode='multiclass')
        self.focal = smp.losses.FocalLoss(mode='multiclass')
        self.ce = nn.CrossEntropyLoss(weight=weights)
        
    def forward(self, y_pred, y_true):
        return self.dice(y_pred, y_true) + self.focal(y_pred, y_true) + self.ce(y_pred, y_true)

def main():
    IMG_PATH = r"C:\PROJECT5\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Color_Images"
    MASK_PATH = r"C:\PROJECT5\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Segmentation"
    
    train_dataset = FalconOffRoadDataset(IMG_PATH, MASK_PATH, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"--- Dataset Loaded: {len(train_dataset)} images ---")

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = RobustLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE).long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 20 == 0: 
                print(f" Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
        print(f"✅ Avg Loss: {epoch_loss/len(train_loader):.4f}")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"falcon_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "falcon_donkey_model.pth")
    print("--- Training Finished ---")

if __name__ == "__main__":
    main()