import os
import torch
import numpy as np
import sklearn

# --- ALLOWLIST THE BLOCKED GLOBALS ---
torch.serialization.add_safe_globals([
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    sklearn.preprocessing._label.LabelEncoder
])


import torch.optim as optim
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Import custom modules
from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherDNN, StudentMLP
from models.trust_gate import TGKD_TrustModule
from scripts.train import TGKD_Trainer

def main(args):
    torch.cuda.empty_cache()
    
    # --- 1. CONFIGURATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running TGKD-IoT Framework on: {device}")
    
    # Ensure Save Directory Exists
    SAVE_DIR = "/content/drive/MyDrive/TA_KD_Research/Checkpoints/"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # --- 2. DATA PREPARATION ---
    print("\n[1/4] Loading and Preprocessing CIC-IoT-2023 Dataset...")
    X_train, X_test, y_train, y_test, X_benign, le = preprocess_iot_data(args.data_path)
    
    # Note: Using args.batch_size from command line
    train_loader, test_loader = get_dataloaders(
        X_train, X_test, y_train, y_test, 
        batch_size=args.batch_size
    )
    
    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)

    # --- 3. MODEL & MODULE INITIALIZATION ---
    print("[2/4] Initializing Models & Trust Module...")
    
    # Initialize Teacher and Load Weights from Dictionary Checkpoint
    teacher = TeacherDNN(input_dim, num_classes).to(device)
    print(f"📂 Loading Teacher from: {args.teacher_path}")
    
    
    checkpoint = torch.load(args.teacher_path, map_location=device)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    
    # This ensures the Student and Teacher always use the same class IDs
    if 'le' in checkpoint:
        le = checkpoint['le']
        print(f"✅ LabelEncoder synced. Number of classes: {len(le.classes_)}")
    
    teacher.eval() # Teacher is always in eval mode
    print("✅ Teacher weights successfully extracted from checkpoint.")
    # Initialize Student
    student = StudentMLP(input_dim, num_classes).to(device)
    
    # Initialize Trust Module
    trust_module = TGKD_TrustModule(contamination=0.05, w=[0.4, 0.4, 0.2])
    print("🛠️ Fitting Anomaly Detector on Benign Traffic...")
    trust_module.fit_anomaly_detector(X_benign)

    # Initialize Trainer (Passing args to control trust usage)
    trainer = TGKD_Trainer(
        student=student, 
        teacher=teacher, 
        trust_module=trust_module, 
        alpha=args.alpha, 
        beta=args.beta,
        accumulation_steps=1 # Simplified for large GPU runs
    )

    optimizer = optim.Adam(student.parameters(), lr=1e-3)

    # --- 4. TRAINING LOOP ---
    print(f"\n[3/4] Starting Trust-Gated Knowledge Distillation...")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Trust Gate: {args.use_trust}")
    
    best_loss = float('inf')
    results_history = []
    csv_path = os.path.join(SAVE_DIR, "experiment_results.csv")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Training
        epoch_loss = trainer.train_epoch(train_loader, optimizer, device)
        
        # Evaluation
        metrics = trainer.evaluate(
            test_loader, 
            device, 
            epoch=epoch, 
            total_epochs=args.epochs, 
            label_names=le.classes_
        )

        # Save Best Model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'loss': epoch_loss,
                'accuracy': metrics['accuracy']
            }, os.path.join(SAVE_DIR, "student_best.pth"))
            print(f"⭐ New Best Student Model Saved!")

        # Log results
        epoch_data = {
            "epoch": epoch,
            "loss": epoch_loss,
            "accuracy": metrics['accuracy'],
            "f1_score": metrics['f1'],
            "precision": metrics['precision'],
            "recall": metrics['recall']
        }
        results_history.append(epoch_data)
        pd.DataFrame(results_history).to_csv(csv_path, index=False)
        
        print(f"Avg Loss: {epoch_loss:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    print(f"\n[4/4] Training Complete. Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGKD for IoT Anomaly Detection")
    
    # Data & Model Paths
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--teacher_path', type=str, required=True, help='Path to teacher .pth file')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--alpha', type=float, default=0.5, help='KD weight')
    parser.add_argument('--beta', type=float, default=0.1, help='Feature Alignment weight')
    
    # Architecture & Research Variables
    parser.add_argument('--student_type', choices=['tiny', 'small', 'medium'], default='small')
    parser.add_argument('--use_trust', action='store_true', help='Enable Trust-Gated Distillation')
    
    args = parser.parse_args()
    
    # Run main
    main(args)