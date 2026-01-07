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

def get_model(num_classes=8):

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def train_epoch(model, loader, crit, optm, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (frames, boxes, actions, group_labels) in enumerate(pbar):

        middle_idx = frames.shape[1] // 2
        inputs = frames[:, middle_idx, :, :, :].to(device)
        labels = group_labels.to(device)

        optm.zero_grad()

        outputs = model(inputs)
        loss = crit(outputs, labels)

        loss.backward()
        optm.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': running_loss / (batch_idx + 1), 'acc': 100 * correct / total})
    
    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, crit, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, boxes, actions, group_labels in tqdm(loader, desc="Validating"):
            # Select middle frame
            middle_idx = frames.shape[1] // 2
            inputs = frames[:, middle_idx, :, :, :].to(device)
            labels = group_labels.to(device)

            outputs = model(inputs)
            loss = crit(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100 * correct / total


def main():
    # --- Hyperparameters ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running on: {DEVICE}")

    # --- Data Setup ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
    ])

    train_dataset = VolleyDataset(
        data_root=Config.DATA_ROOT,
        tracking_root=Config.TRACKING_ROOT,
        split='train',
        resize_dims=(224, 224),     # ensures input fits ResNet
        return_crops=False,         # ensures we get the full image
        transform=transform,
        print_logs=False
    )
    
    val_dataset = VolleyDataset(
        data_root=Config.DATA_ROOT,
        tracking_root=Config.TRACKING_ROOT,
        split='val', 
        resize_dims=(224, 224),
        return_crops=False,
        transform=transform,
        print_logs=False
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # --- Model Setup ---
    model = get_model(num_classes=8)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    # --- Training Loop ---
    best_acc = 0.0
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            if Config.SAVE:
                torch.save(model.state_dict(), "b1_resnet50_best.pth")
                print("B1 --> Best model saved.")
    
    print(f"\nFinal Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
