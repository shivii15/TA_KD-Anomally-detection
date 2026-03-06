import torch
import torch.nn as nn
import torch.optim as optim

def train_baseline(model, loader, epochs=10, lr=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("🚀 Training Baseline Student...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")
    return model