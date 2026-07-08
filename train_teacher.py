import argparse
import torch
import torch.optim as optim
import torch.nn as nn
#from data.data_loader import preprocess_iot_data, get_dataloaders
from data.data_loader import load_dataset, get_dataloaders
from datetime import datetime
from models.model import (
    TeacherDNN,
    TeacherResNet,
    TeacherTransformer,
    TeacherLSTM
)
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Train the Expert Teacher Model")
    
    # --- ARGUMENTS ---
    parser.add_argument(
        "--dataset",
        default="cic",
        choices=["cic", "nbaiot"]
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="dnn",
        choices=[
            "dnn",
            "resnet",
            "transformer",
            "lstm"
        ],
        help="Teacher architecture"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Dataset directory"
    )
    #parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    #parser.add_argument('--save_path', type=str, default="models/teacher_best.pth")
    parser.add_argument('--num_parts', type=int, default=5, 
                    help='Number of CSV parts to load (use -1 for ALL parts)')
    parser.add_argument('--save_scaler', type=str, default='models/scaler.pkl', 
                    help='Where to save the StandardScaler for the Student to use later')
    
    args = parser.parse_args()

    if args.dataset == "cic" and args.data_path is None:
        raise ValueError(
            "--data_path is required when using the CIC-IoT dataset."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    save_path = (
        f"models/Teacher_{args.teacher}_{args.dataset}_{timestamp}.pth"
    )

    log_path = (
        f"logs/Teacher_{args.teacher}_{args.dataset}_{timestamp}.json"
    )

    # --- SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure the models directory exists
    #os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 1. Load Data
    #print(f"📂 Loading data from: {args.data_path}")
    
    print(f"\nDataset : {args.dataset}")

    if args.data_path:
        print(f"Data Path : {args.data_path}")
    
    #X_train, y_train, le = preprocess_iot_data(args.data_path, num_parts=5)
    #train_loader, test_loader = get_dataloaders(X_train, y_train, batch_size=args.batch_size)
# Call the updated loader with correct variable mapping
    X_processed, y_processed, le, scaler = load_dataset(args)    
    train_loader, val_loader = get_dataloaders(X_processed, y_processed, batch_size=args.batch_size)
    
    print(f"🏷️ Class Labels: {list(le.classes_)}")


    # 2. Initialize Teacher
    input_dim = X_processed.shape[1]
    num_classes = len(le.classes_)

    print("\nDataset Summary")
    print("-" * 40)
    print(f"Samples      : {len(X_processed):,}")
    print(f"Features     : {input_dim}")
    print(f"Classes      : {num_classes}")
    print(f"Batch Size   : {args.batch_size}")
    print("-" * 40)

    #model = TeacherDNN(input_dim=input_dim, num_classes=num_classes).to(device)
    if args.teacher == "dnn":

        model = TeacherDNN(
            input_dim,
            num_classes
        )

    elif args.teacher == "resnet":

        model = TeacherResNet(
            input_dim,
            num_classes
        )

    elif args.teacher == "transformer":

        model = TeacherTransformer(
            input_dim,
            num_classes
        )

    elif args.teacher == "lstm":

        model = TeacherLSTM(
            input_dim,
            num_classes
        )

    else:

        raise ValueError(
            f"Unknown teacher : {args.teacher}"
        )

    model = model.to(device)
    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable Parameters : {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()



    # 3. Training Loop
    #print(f"🚀 Training Teacher for {args.epochs} epochs on {device}...")
    print("\nTeacher Configuration")
    print("-" * 40)
    print(f"Teacher      : {args.teacher}")
    print(f"Dataset      : {args.dataset}")
    print(f"Device       : {device}")
    print(f"Epochs       : {args.epochs}")
    print("-" * 40)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # -----------------------------
        # Training
        # -----------------------------
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

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()

        val_loss = 0

        with torch.no_grad():

            for batch_x, batch_y in val_loader:

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits, _ = model(batch_x)

                loss = criterion(logits, batch_y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        #print(f"Epoch {epoch:02d}/{args.epochs} | Avg Loss: {avg_loss:.4f}")
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )
        # --- SMART SAVING ---
        #if avg_loss < best_loss:
        if val_loss < best_loss:
            best_loss = val_loss
        torch.save({
            'dataset': args.dataset,
            'model_state_dict': model.state_dict(),
            'le': le,
            'scaler': scaler,
            'input_dim': input_dim,
            'num_classes': num_classes
        }, save_path)

        print(f"💾 Saved improved model to {save_path}")

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)

        with open(log_path, "w") as f:
            json.dump(history, f, indent=4)

    print(f"✅ Teacher training complete. Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()