import torch
import torch.nn as nn
import torch.optim as optim
from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherResNet
import argparse
import os
from datetime import datetime

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Standardizing the forward pass call
            logits, _ = model(batch_x) 
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description="TGKD Phase 1: SOTA Teacher Training")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--num_parts', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
    parser.add_argument('--save_path', type=str, default="models/teacher_resnet_best.pth")
    parser.add_argument('--version', type=str, default="v2.1-SOTA-ResNet")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Experiment: {args.version} | Device: {device}")

    # 1. Load Data
    X, y, le, scaler = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    train_loader, val_loader = get_dataloaders(X, y, batch_size=args.batch_size)

    # 2. Initialize Teacher
    input_dim = X.shape[1]
    num_classes = len(le.classes_)
    model = TeacherResNet(input_dim, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # --- SOTA UPGRADE: AdamW ---
    # Decouples weight decay from the gradient update for better ResNet convergence
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    
    # --- SOTA UPGRADE: CosineAnnealingLR ---
    # Smoothly decays LR to a minimum (1e-6) over the total epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 3. Training Loop Variables
    best_val_loss = float('inf')
    epochs_no_improve = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    args.save_path = f"models/Teacher_{args.version}_{timestamp}.pth"
    
    print(f"🟢 Training started at {datetime.now().strftime('%H:%M:%S')}")
    
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
        
        # Step the scheduler every epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch}/{args.epochs}] Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        # Early Stopping & Best Model Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'classes': le.classes_, # Saving labels strictly for Student sync
                'metadata': {
                    'version': args.version,
                    'input_dim': input_dim,
                    'num_classes': num_classes,
                    'final_val_acc': val_acc,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(checkpoint, args.save_path)
            print(f"⭐ New Best Model Saved (Acc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"🛑 Early stopping triggered. No improvement for {args.patience} epochs.")
                break

if __name__ == "__main__":
    main()