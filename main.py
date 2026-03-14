import os
import torch
import torch.optim as optim
import numpy as np

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Import your custom modules
from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherDNN, StudentMLP
from trust_gate import TGKD_TrustModule
from scripts.train import TGKD_Trainer
from scripts.attack import test_robustness

def main():
    torch.cuda.empty_cache()
    
    # --- 1. CONFIGURATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running TGKD-IoT Framework on: {device}")
    
    # Paths and Hyperparameters
    DATA_PATH = "data/CICIoT2023_xxsmall.csv"
    
    # Physical vs Effective Batch Size
    # Target is 1024 (per your paper). If GPU is small, load 64 at a time.
    PHYSICAL_BATCH = 64 
    ACCUMULATION_STEPS = 16 # 64 * 16 = 1024
    
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    ALPHA = 0.5  # Distillation Weight
    BETA = 0.1   # Feature Alignment Weight

    # --- 2. DATA PREPARATION ---
    print("Loading and Preprocessing CIC-IoT-2023 Dataset...")
    X_train, X_test, y_train, y_test, X_benign = preprocess_iot_data(DATA_PATH)
    
    train_loader, test_loader = get_dataloaders(
        X_train, X_test, y_train, y_test, 
        batch_size=PHYSICAL_BATCH
    )
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # --- 3. MODEL & MODULE INITIALIZATION ---
    # Teacher (ResNet-style DNN) and Student (Tiny MLP)
    teacher = TeacherDNN(input_dim, num_classes).to(device)
    student = StudentMLP(input_dim, num_classes).to(device)
    
    # Initialize the Trust Module (Isolation Forest + Confidence)
    # Calibrate only on benign data
    trust_module = TGKD_TrustModule(contamination=0.05, w=[0.4, 0.4, 0.2])
    trust_module.fit_anomaly_detector(X_benign)

    # Initialize the High-Efficiency Trainer (with AMP Autocast)
    trainer = TGKD_Trainer(
        student=student, 
        teacher=teacher, 
        trust_module=trust_module, 
        alpha=ALPHA, 
        beta=BETA,
        accumulation_steps=ACCUMULATION_STEPS
    )

    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)

    # --- 4. TRAINING LOOP ---
    print(f"\nStarting Trust-Gated Knowledge Distillation...")
    print(f"Effective Batch Size: {PHYSICAL_BATCH * ACCUMULATION_STEPS}")
    
    for epoch in range(1, EPOCHS + 1):
        loss = trainer.train_epoch(train_loader, optimizer, device)