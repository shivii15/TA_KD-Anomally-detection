import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherDNN
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Train the Expert Teacher Model")
    
    # --- ARGUMENTS ---
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_path', type=str, default="models/teacher_best.pth")
    parser.add_argument('--num_parts', type=int, default=5, 
                    help='Number of CSV parts to load (use -1 for ALL parts)')
    parser.add_argument('--save_scaler', type=str, default='models/scaler.pkl', 
                    help='Where to save the StandardScaler for the Student to use later')
    
    args = parser.parse_args()

    # --- SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure the models directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    # 1. Load Data
    print(f"📂 Loading data from: {args.data_path}")
    #X_train, y_train, le = preprocess_iot_data(args.data_path, num_parts=5)
    #train_loader, test_loader = get_dataloaders(X_train, y_train, batch_size=args.batch_size)
# Call the updated loader with correct variable mapping
    X_processed, y_processed, le, scaler = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    
    train_loader, val_loader = get_dataloaders(X_processed, y_processed, batch_size=args.batch_size)
    
    print(f"🏷️ Class Labels: {list(le.classes_)}")


    # 2. Initialize Teacher
    input_dim = X_processed.shape[1]
    num_classes = len(le.classes_)
    model = TeacherDNN(input_dim=input_dim, num_classes=num_classes).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()



    # 3. Training Loop
    print(f"🚀 Training Teacher for {args.epochs} epochs on {device}...")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x) # Your model returns (logits, features)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d}/{args.epochs} | Avg Loss: {avg_loss:.4f}")

        # --- SMART SAVING ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'le': le,
                'scaler':scaler
                'input_dim': input_dim,
                'num_classes': num_classes
            }, args.save_path)
            print(f"💾 Saved improved model to {args.save_path}")

    print(f"✅ Teacher training complete. Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()