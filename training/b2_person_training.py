import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

from data.config import Config
from data.PersonDataset import get_person_loader

def get_person_model(num_classes=9, dropout=0.5):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_ftrs, num_classes)
    )

    return model

def train_one_epoch(model, loader, crit, optm, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (crops, labels) in enumerate(pbar):
        
        inputs = crops.to(device)
        targets = labels.to(device)

        optm.zero_grad()

        outputs = model(inputs)
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
        for crops, labels in tqdm(loader, desc="Validating"):
            inputs = crops.to(device)
            targets = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return running_loss / len(loader), 100 * correct / total

def main():
    # --- Hyperparameters ---
    BATCH_SIZE = 32
    LR = 3e-4
    EPOCHS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running on: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LR}")

    # --- Data Setup ---
    print("\nInitializing Datasets...")
    
    train_loader = get_person_loader(
        split='train',
        batch_size=BATCH_SIZE,
        seq=False,
        pkl_path=Config.PKL_PATH,
        print_logs=True,
        only_target=False
    )
    
    val_loader = get_person_loader(
        split='val',
        batch_size=BATCH_SIZE,
        seq=False,
        pkl_path=Config.PKL_PATH,
        print_logs=False,
        only_target=False
    )

    # --- Model Setup ---
    num_classes = len(Config.PERSON_LABELS)
    model = get_person_model(num_classes=num_classes, dropout=0.5) 
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # --- Training Loop ---
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print('='*60)
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            if Config.SAVE:
                torch.save(model.state_dict(), "b2_resnet50_person_best.pth")
                print("--> Best model saved.")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print('='*60)
    print(f"Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()