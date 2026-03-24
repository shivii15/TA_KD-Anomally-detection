import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
from datetime import datetime
from tqdm import tqdm

# Import your data loader and all architectures
from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherResNet, TeacherTransformer, TeacherLSTM, TeacherDNN

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
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
    parser = argparse.ArgumentParser(description="Universal Teacher Training Engine")
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['resnet', 'transformer', 'lstm', 'dnn'],
                        help="Choose the architecture to train")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--num_parts', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 1. Setup Logging and Save Paths
    save_dir = f"models/teachers/{args.model_type}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    model_name = f"Teacher_{args.model_type.upper()}_SOTA_{timestamp}"
    full_save_path = os.path.join(save_dir, f"{model_name}.pth")
    log_path = f"logs/{model_name}_history.json"

    # 2. Data Preparation
    X, y, le, scaler = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    train_loader, val_loader = get_dataloaders(X, y, batch_size=args.batch_size)
    num_classes = len(le.classes_)
    input_dim = X.shape[1]

    # 3. Model Selection
    if args.model_type == 'resnet':
        model = TeacherResNet(input_dim, num_classes).to(device)
    elif args.model_type == 'transformer':
        model = TeacherTransformer(input_dim, num_classes).to(device)
    elif args.model_type == 'lstm':
        model = TeacherLSTM(input_dim, num_classes).to(device)
    else:
        model = TeacherDNN(input_dim, num_classes).to(device)

    # 4. Optimizer & SOTA Scheduler (Cosine Annealing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 5. History Tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

if __name__ == "__main__":
    main()