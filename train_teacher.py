import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherDNN
import os

def main():
    parser = argparse.ArgumentParser(description="Train the Expert Teacher Model")
    
    # --- ARGUMENTS ---
    parser.add_argument('--data_path', type=str, default="data/CICIoT2023_small.csv", help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default="models/teacher_best.pth", help='Path to save model')
    
    args = parser.parse_args()

    # --- SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # 1. Load Data
    print(f"📂 Loading data from: {args.data_path}")
    X_train, X_test, y_train, y_test, _, le = preprocess_iot_data(args.data_path)
    train_loader, _ = get_dataloaders(X_train, X_test, y_train, y_test, batch_size=args.batch_size)

    # 2. Initialize Teacher
    model = TeacherDNN(input_dim=X_train.shape[1], num_classes=len(le.classes_)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    print(f"🚀 Training Teacher for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}/{args.epochs} | Avg Loss: {total_loss/len(train_loader):.4f}")

    # 4. Save
    torch.save(model.state_dict(), args.save_path)
    print(f"✅ Teacher training complete. Saved to: {args.save_path}")

if __name__ == "__main__":
    main()