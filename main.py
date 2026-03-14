import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm  # For the progress bar

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Import your custom modules
from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherDNN, StudentMLP
from models.trust_gate import TGKD_TrustModule
from scripts.train import TGKD_Trainer
from scripts.attack import test_robustness

def main():
    torch.cuda.empty_cache()
    
    # --- 1. CONFIGURATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running TGKD-IoT Framework on: {device}")
    
    # Paths
    DATA_PATH = "data/CICIoT2023_xxsmall.csv"
    # Change this to your preferred Drive folder
    SAVE_DIR = "/content/drive/MyDrive/TA_KD_Research/Checkpoints/"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Hyperparameters
    PHYSICAL_BATCH = 64 
    ACCUMULATION_STEPS = 16 
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    ALPHA = 0.5  
    BETA = 0.1   

    # --- 2. DATA PREPARATION ---
    print("\n[1/4] Loading and Preprocessing CIC-IoT-2023 Dataset...")
    X_train, X_test, y_train, y_test, X_benign = preprocess_iot_data(DATA_PATH)
    
    train_loader, test_loader = get_dataloaders(
        X_train, X_test, y_train, y_test, 
        batch_size=PHYSICAL_BATCH
    )
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # --- 3. MODEL & MODULE INITIALIZATION ---
    print("[2/4] Initializing Models & Trust Module...")
    teacher = TeacherDNN(input_dim, num_classes).to(device)
    student = StudentMLP(input_dim, num_classes).to(device)
    
    trust_module = TGKD_TrustModule(contamination=0.05, w=[0.4, 0.4, 0.2])
    trust_module.fit_anomaly_detector(X_benign)

    trainer = TGKD_Trainer(
        student=student, 
        teacher=teacher, 
        trust_module=trust_module, 
        alpha=ALPHA, 
        beta=BETA,
        accumulation_steps=ACCUMULATION_STEPS
    )

    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)

    # --- 4. TRAINING LOOP WITH PROGRESS BAR ---
    print(f"\n[3/4] Starting Trust-Gated Knowledge Distillation...")
    print(f"Effective Batch Size: {PHYSICAL_BATCH * ACCUMULATION_STEPS}")
    
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # We wrap the trainer inside a progress bar manually if trainer.train_epoch doesn't have one
        # Or if trainer.train_epoch uses tqdm internally, it will show up here.
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        # Using a tqdm wrapper for visual feedback
        epoch_loss = trainer.train_epoch(train_loader, optimizer, device)
        
        print(f"Average Epoch Loss: {epoch_loss:.4f}")

        # --- 5. SAVE MODEL CHECKPOINTS ---
        # Save every epoch as 'latest'
        torch.save(student.state_dict(), os.path.join(SAVE_DIR, "student_latest.pth"))
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(student.state_dict(), os.path.join(SAVE_DIR, "student_best.pth"))
            print(f"⭐ New Best Model Saved to Drive!")

    print(f"\n[4/4] Training Complete. Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()