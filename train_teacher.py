import torch
import torch.optim as optim
import torch.nn as nn
from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherDNN
import os

def train_teacher():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH = "data/CICIoT2023_xxsmall.csv" # Ensure path is correct
    SAVE_PATH = "models/teacher_best.pth"
    os.makedirs("models", exist_ok=True)

    # 1. Load Data
    X_train, X_test, y_train, y_test, X_benign, le = preprocess_iot_data(DATA_PATH)
    train_loader, test_loader = get_dataloaders(X_train, X_test, y_train, y_test, batch_size=1024)

    # 2. Initialize Teacher
    model = TeacherDNN(input_dim=X_train.shape[1], num_classes=len(le.classes_)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    print("🚀 Training Teacher Model...")
    for epoch in range(1, 11): # 10 Epochs is usually enough for a strong teacher
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
        
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")

    # 4. Save
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Teacher saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_teacher()