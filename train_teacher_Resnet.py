import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
from datetime import datetime
from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherResNet

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits, _ = model(batch_x) 
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    return total_loss / len(loader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description="SOTA Teacher Training with Auto-Logging")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--num_parts', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_name', type=str, default="ResNet_Teacher_Final_PhD")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Versioning and Paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    full_save_path = f"models/{args.save_name}_{timestamp}.pth"
    log_path = f"models/{args.save_name}_{timestamp}_logs.json"
    os.makedirs('models', exist_ok=True)

    # 2. Load Data
    X, y, le, scaler = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    train_loader, val_loader = get_dataloaders(X, y, batch_size=args.batch_size)

    # 3. Initialize Model
    model = TeacherResNet(X.shape[1], len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Training Loop with History Tracking
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"🟢 Training: {full_save_path}")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x) 
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log Metrics
        current_lr = optimizer.param_groups[0]['lr']
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        scheduler.step()

        print(f"Epoch [{epoch}/{args.epochs}] Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save({'model_state_dict': model.state_dict(), 'classes': le.classes_}, full_save_path)
            # Save logs every time we find a better model
            with open(log_path, 'w') as f:
                json.dump(history, f)
            print(f"⭐ Best Model & Logs Saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"🛑 Early stopping triggered.")
                break

if __name__ == "__main__":
    main()