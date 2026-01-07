import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

from data.config import Config
from data.data_loader import VolleyDataset

def get_person_model(num_classes=9):

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def train_one_epoch(model, loader, crit, optm, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (crops, boxes, actions, group_labels) in enumerate(pbar):

        middle_idx = crops.shape[1] // 2
        inputs = crops[:, middle_idx, :, :, :].to(device)
        targets = actions[:, middle_idx, :].to(device)

        # New Shape: (B * 12, C, H, W)
        b, p, c, h, w = inputs.shape
        inputs = inputs.view(b * p, c, h, w)
        targets = targets.view(b * p)

        optm.zero_grad()

        outputs = model(inputs) # outputs: (B*12, 9)
        loss = crit(outputs, targets)

        loss.backward()
        optm.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        pbar.set_postfix({'loss': running_loss / (batch_idx + 1), 'acc': 100 * correct / total})

    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for crops, boxes, actions, group_labels in tqdm(loader, desc="Validating"):
            # Select middle frame
            middle_idx = crops.shape[1] // 2
            
            inputs = crops[:, middle_idx, :, :, :].to(device)
            targets = actions[:, middle_idx, :].to(device)

            # Flatten
            b, p, c, h, w = inputs.shape
            inputs = inputs.view(b * p, c, h, w)
            targets = targets.view(b * p)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return running_loss / len(loader), 100 * correct / total

def main():
    # --- Hyperparameters ---
    BATCH_SIZE = 1 # Reduced since we flatten (B, 12), BATCH_SIZE * P (12) is too big for our GPU's mem
    LR = 1e-4
    EPOCHS = 15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running on: {DEVICE}")

    # --- Data Setup ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
    ])

    print("Initializing Datasets...")
    train_dataset = VolleyDataset(
        data_root=Config.DATA_ROOT,
        tracking_root=Config.TRACKING_ROOT,
        split='train',
        resize_dims=(720, 1280), # Full frame size
        crop_size=(224, 224),    # Player crop size
        return_crops=True,       # Enable cropping mode
        transform=transform,
        print_logs=False
    )
    
    val_dataset = VolleyDataset(
        data_root=Config.DATA_ROOT,
        tracking_root=Config.TRACKING_ROOT,
        split='val',
        resize_dims=(720, 1280),
        crop_size=(224, 224),
        return_crops=True,
        transform=transform,
        print_logs=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # --- Model Setup ---
    model = get_person_model(num_classes=9) 
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # --- Training Loop ---
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            if Config.SAVE:
                torch.save(model.state_dict(), "b2_resnet50_person_best.pth")
                print("--> Best model saved.")

    print(f"\nFinal Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
