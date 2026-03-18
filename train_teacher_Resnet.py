import torch
import torch.nn as nn
import torch.optim as optim
from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherResNet  # Ensure this is in your model.py
import argparse
import os
import time
from datetime import datetime

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description="TGKD Phase 1: High-Capacity Teacher Training")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--num_parts', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=7, help="Early stopping patience")
    parser.add_argument('--save_path', type=str, default="models/teacher_resnet_v2.pth")
    parser.add_argument('--version', type=str, default="v2.0-ResNet")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Experiment: {args.version} | Device: {device}")

    # 1. Load Data (Unpacking 4 values as per our fix)
    print(f"📂 Loading and Preprocessing {args.num_parts if args.num_parts != -1 else 'all'} parts...")
    X, y, le, scaler = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    train_loader, val_loader = get_dataloaders(X, y, batch_size=args.batch_size)

    # 2. Initialize Teacher (ResNet variant)
    input_dim = X.shape[1]
    num_classes = len(le.classes_)
    model = TeacherResNet(input_dim, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # 3. Training with Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"🟢 Training started at {datetime.now().strftime('%H:%M:%S')}")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step the LR scheduler
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch}/{args.epochs}] Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Early Stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save strictly with all PhD metadata
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'le': le,
                'scaler': scaler,
                'metadata': {
                    'version': args.version,
                    'input_dim': input_dim,
                    'num_classes': num_classes,
                    'final_val_acc': val_acc,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            torch.save(checkpoint, args.save_path)
            print(f"⭐ Best Model Saved (Acc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"🛑 Early stopping triggered. No improvement for {args.patience} epochs.")
                break

if __name__ == "__main__":
    main()