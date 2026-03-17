import os
import torch
import torch.optim as optim
import numpy as np
import sklearn
import argparse
import pandas as pd
import joblib  # Added for loading Teacher's state
from tqdm import tqdm
import warnings
import logging
from datetime import datetime

# --- 1. SUPPRESS NOISY WARNINGS ---
warnings.filterwarnings("ignore", message=".*autocast.*")
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
logging.getLogger("torch").setLevel(logging.ERROR)

# --- 2. ENVIRONMENT & CUSTOM MODULES ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherDNN, StudentMLP
from models.trust_gate import TGKD_TrustModule
from scripts.train import TGKD_Trainer

def main(args):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 3. RUN ID & PATHS ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_id = f"TGKD_{args.student_type}_a{args.alpha}_b{args.beta}_{timestamp}"
    SAVE_DIR = os.path.join(os.getcwd(), "outputs", run_id)
    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(SAVE_DIR, f"metrics_{run_id}.csv")

    print(f"🚀 Experiment ID: {run_id}")
    print(f"💻 Running on: {device}")
    
    # --- 4. DATA PREPARATION ---
    print("\n[1/4] Loading and Preprocessing Dataset...")
    # Updated to use the num_parts argument for server flexibility
    #X_train, y_train, le = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    # Unpack 4 values instead of 3
    X_train, y_train, le, _ = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    train_loader, test_loader = get_dataloaders(X_train, y_train, batch_size=args.batch_size)
    
    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)

    # --- 5. MODEL INITIALIZATION ---
    print("[2/4] Initializing Models & Trust Module...")
    
    # Teacher setup
    teacher = TeacherDNN(input_dim, num_classes).to(device)
    print(f"📂 Loading Teacher: {os.path.basename(args.teacher_path)}")
    checkpoint = torch.load(args.teacher_path, map_location=device, weights_only=False)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.eval()

    # Student setup
    student = StudentMLP(input_dim, num_classes).to(device)
    
    # --- 6. TRUST MODULE SETUP ---
    # We extract Benign traffic specifically from the training set for the Trust Module
    # In CIC-IoT-2023, 'BenignTraffic' is the typical label for normal data
    if 'BenignTraffic' in le.classes_:
        benign_idx = list(le.classes_).index('BenignTraffic')
        X_benign = X_train[y_train == benign_idx]
    else:
        # Fallback if label is different
        X_benign = X_train[:5000] 
        
    trust_module = TGKD_TrustModule(contamination=0.05, w=[0.4, 0.4, 0.2])
    print("🛠️ Fitting Anomaly Detector on Benign Traffic...")
    trust_module.fit_anomaly_detector(X_benign)

    # --- 7. TRAINING CONFIGURATION ---
    trainer = TGKD_Trainer(
        student=student, 
        teacher=teacher, 
        trust_module=trust_module, 
        alpha=args.alpha, 
        beta=args.beta
    )
    optimizer = optim.Adam(student.parameters(), lr=1e-3)

    # --- 8. DISTILLATION LOOP ---
    print(f"\n[3/4] Starting Distillation...")
    best_loss = float('inf')
    results_history = []

    for epoch in range(1, args.epochs + 1):
        epoch_loss = trainer.train_epoch(train_loader, optimizer, device)
        
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
                'run_id': run_id,
                'model_state_dict': student.state_dict(),
                'le': le
            }, os.path.join(SAVE_DIR, f"best_student_{run_id}.pth"))
            print(f"⭐ Epoch {epoch:02d}: New Best Model Saved")

        # Logging
        epoch_data = {
            "epoch": epoch, "loss": round(epoch_loss, 5),
            "accuracy": round(metrics['accuracy'], 5), "f1": round(metrics['f1'], 5)
        }
        results_history.append(epoch_data)
        pd.DataFrame(results_history).to_csv(csv_path, index=False)
        
        print(f"📊 Ep {epoch:02d} Summary: Loss={epoch_loss:.4f} | Acc={metrics['accuracy']:.4f}")

    print(f"\n[4/4] Complete! Artifacts saved in: {SAVE_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGKD for IoT Anomaly Detection")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--student_type', choices=['tiny', 'small', 'medium'], default='small')
    parser.add_argument('--num_parts', type=int, default=2) # Crucial for server control
    
    args = parser.parse_args()
    main(args)