import sys
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.config import Config
from data.helpers import compute_metrics, plot_confusion_matrix
from data.GroupDataset import get_group_loader


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=8, dropout=0.2):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


def train_epoch(model, loader, crit, optm, device, scaler, writer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optm.zero_grad()
        
        # Mixed precision training
        with autocast(dtype=torch.float16):
            outputs = model(inputs)
            loss = crit(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optm)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Log to tensorboard every 50 batches
        if batch_idx % 50 == 0:
            step = epoch * len(loader) + batch_idx
            writer.add_scalar('Loss/train/batch', loss.item(), step)
            writer.add_scalar('Accuracy/train/batch', 100 * correct / total, step)

        pbar.set_postfix({
            'loss': f'{running_loss / (batch_idx + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, crit, device, return_predictions=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = crit(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    
    if return_predictions:
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return avg_loss, accuracy, f1, np.array(all_preds), np.array(all_labels)
    
    return avg_loss, accuracy


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED if hasattr(Config, 'SEED') else 42)
    
    # Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {DEVICE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LR}")
    print(f"Epochs: {Config.EPOCHS}\n")

    # TensorBoard
    writer = SummaryWriter(log_dir='runs/b1_experiment')

    # Data Loaders
    train_loader = get_group_loader(
        split='train',
        batch_size=Config.BATCH_SIZE,
        seq=False,
        crops=False,
        print_logs=False,
        only_target=False  # All frames
    )
    
    val_loader = get_group_loader(
        split='val',
        batch_size=Config.BATCH_SIZE,
        seq=False,
        crops=False,
        print_logs=False,
        only_target=False
    )

    # Model Setup
    num_classes = len(Config.GROUP_LABELS)
    model = get_model(num_classes=num_classes, dropout=0.2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    
    # Add weight decay and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LR, 
        weight_decay=Config.WEIGHT_DECAY if hasattr(Config, 'WEIGHT_DECAY') else 1e-4
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    scaler = GradScaler()

    # Early stopping
    patience = Config.PATIENCE if hasattr(Config, 'PATIENCE') else 5
    patience_counter = 0
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler, writer, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        if writer:
            writer.add_scalar('Loss/train/epoch', train_loss, epoch)
            writer.add_scalar('Accuracy/train/epoch', train_acc, epoch)
            writer.add_scalar('Loss/val/epoch', val_loss, epoch)
            writer.add_scalar('Accuracy/val/epoch', val_acc, epoch)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        scheduler.step(val_acc)
        
        # Early stopping check
        if val_acc > best_val_acc + (Config.MIN_DELTA if hasattr(Config, 'MIN_DELTA') else 0.01):
            best_val_acc = val_acc
            patience_counter = 0
            if Config.SAVE:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, "b1_resnet50_best.pth")
                print("--> Best model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Final Evaluation
    print("\nRunning final evaluation...")
    checkpoint = torch.load("b1_resnet50_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss, val_acc, val_f1, y_pred, y_true = validate(
        model, val_loader, criterion, DEVICE, return_predictions=True
    )
    
    print(f"\nFinal Validation Accuracy: {val_acc:.2f}%")
    print(f"Final Validation F1-Score: {val_f1:.4f}")
    
    # Metrics
    class_names = [k for k, v in sorted(Config.GROUP_LABELS.items(), key=lambda x: x[1])]
    cm, mca = compute_metrics(y_true, y_pred, class_names)
    plot_confusion_matrix(cm, class_names)
    
    # Save history
    np.save('b1_training_history.npy', history)
    
    writer.close()
    print("\nAll results saved!")


if __name__ == "__main__":
    main()