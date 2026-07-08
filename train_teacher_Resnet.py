import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
from datetime import datetime
from tqdm import tqdm  # ✅ New Import
from data.data_loader import load_dataset, get_dataloaders
from models.model import TeacherResNet

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    # Add a mini progress bar for validation too
    val_pbar = tqdm(loader, desc="🔍 Validating", leave=False)
    with torch.no_grad():
        for batch_x, batch_y in val_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits, _ = model(batch_x) 
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    return total_loss / len(loader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description="SOTA Teacher Training with Live Progress")
    parser.add_argument(
        "--dataset",
        default="cic",
        choices=["cic", "nbaiot"]
    )
    #parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Dataset directory (optional for N-BaIoT)"
    )
    parser.add_argument('--num_parts', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_name', type=str, default="Teacher_v2.1-SOTA-ResNet")
    args = parser.parse_args()

    if args.dataset == "cic" and args.data_path is None:
        raise ValueError(
            "--data_path is required when using the CIC-IoT dataset."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Path Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    full_save_path = f"models/{args.save_name}_{timestamp}.pth"
    log_path = f"logs/{args.save_name}_{timestamp}_history.json"

    # 2. Data & Model
    X, y, le, scaler = load_dataset(args)

    print(f"\nDataset : {args.dataset}")

    input_dim = X.shape[1]
    num_classes = len(le.classes_)

    print("\nDataset Summary")
    print("-" * 40)
    print(f"Samples      : {len(X):,}")
    print(f"Features     : {input_dim}")
    print(f"Classes      : {num_classes}")
    print(f"Batch Size   : {args.batch_size}")
    print("-" * 40)

    train_loader, val_loader = get_dataloaders(X, y, batch_size=args.batch_size)
    model = TeacherResNet(
        input_dim,
        num_classes
    ).to(device)        

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable Parameters : {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_val_loss": [],
        "val_acc": [],
        "lr": []
    }
    
    print(f"🟢 Training: {args.save_name}")
    print("\nClasses")

    for c in le.classes_:
        print(" •", c)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        
        # ✅ Wrapped train_loader with tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]", unit="batch")
        
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x) 
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # Update the progress bar with the current loss
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log and step scheduler

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]['lr'])
        history["best_val_loss"].append(best_val_loss)
        scheduler.step()

        print(
            f"Best Validation Loss : {best_val_loss:.4f}"
        )

        # ✅ Robust JSON Save (Prevents empty files)
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=4)

        print(f"📊 Summary: Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            #torch.save({'model_state_dict': model.state_dict(), 'classes': le.classes_}, full_save_path)
            torch.save({
                "dataset": args.dataset,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "le": le,
                "scaler": scaler,
                "input_dim": input_dim,
                "num_classes": num_classes
            }, full_save_path)
            print(f"⭐ Best Model Saved to {full_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"🛑 Early stopping triggered.")
                break

if __name__ == "__main__":
    main()