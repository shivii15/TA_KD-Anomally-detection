import os
import torch
import torch.optim as optim
import numpy as np
import sklearn
import argparse
import pandas as pd
from tqdm import tqdm
import warnings
import logging
from datetime import datetime

# --- 1. SUPPRESS NOISY WARNINGS ---
warnings.filterwarnings("ignore", message=".*autocast.*")
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
logging.getLogger("torch").setLevel(logging.ERROR)

# --- 2. FIX FOR PYTORCH 2.6+ SECURITY ERRORS ---
try:
    from numpy.dtypes import ObjectDType
    torch.serialization.add_safe_globals([
        ObjectDType, 
        np._core.multiarray._reconstruct, 
        np.ndarray, 
        np.dtype,
        sklearn.preprocessing._label.LabelEncoder
    ])
except (ImportError, AttributeError):
    pass

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Import custom modules
from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherDNN, StudentMLP
from models.trust_gate import TGKD_TrustModule
from scripts.train import TGKD_Trainer

def main(args):
    torch.cuda.empty_cache()
    
    # --- 3. GENERATE UNIQUE RUN ID & PATHS ---
    # Example: TGKD_small_a0.5_b0.1_20260315_1430
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_id = f"TGKD_{args.student_type}_a{args.alpha}_b{args.beta}_{timestamp}"

    # Set up directories relative to current path for server compatibility
    BASE_DIR = os.getcwd()
    SAVE_DIR = os.path.join(BASE_DIR, "outputs", run_id)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    csv_path = os.path.join(SAVE_DIR, f"metrics_{run_id}.csv")
    
    # --- 4. CONFIGURATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Experiment ID: {run_id}")
    print(f"💻 Running on: {device}")
    
    # --- 5. DATA PREPARATION ---
    print("\n[1/4] Loading and Preprocessing Dataset...")
    X_train, X_test, y_train, y_test, X_benign, le = preprocess_iot_data(args.data_path)
    
    train_loader, test_loader = get_dataloaders(
        X_train, X_test, y_train, y_test, 
        batch_size=args.batch_size
    )
    
    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)

    # --- 6. MODEL & MODULE INITIALIZATION ---
    print("[2/4] Initializing Models & Trust Module...")
    
    teacher = TeacherDNN(input_dim, num_classes).to(device)
    print(f"📂 Loading Teacher: {os.path.basename(args.teacher_path)}")
    
    checkpoint = torch.load(args.teacher_path, map_location=device, weights_only=False)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    
    if 'le' in checkpoint:
        le = checkpoint['le']
        print(f"✅ LabelEncoder synced. Classes: {len(le.classes_)}")
    
    teacher.eval()
    student = StudentMLP(input_dim, num_classes).to(device)
    
    trust_module = TGKD_TrustModule(contamination=0.05, w=[0.4, 0.4, 0.2])
    print("🛠️ Fitting Anomaly Detector on Benign Traffic...")
    trust_module.fit_anomaly_detector(X_benign)

    trainer = TGKD_Trainer(
        student=student, 
        teacher=teacher, 
        trust_module=trust_module, 
        alpha=args.alpha, 
        beta=args.beta
    )

    optimizer = optim.Adam(student.parameters(), lr=1e-3)

    # --- 7. TRAINING LOOP ---
    print(f"\n[3/4] Starting Distillation...")
    
    best_loss = float('inf')
    results_history = []

    for epoch in range(1, args.epochs + 1):
        # Training (Silent mode - internal prints should be removed in scripts/train.py)
        epoch_loss = trainer.train_epoch(train_loader, optimizer, device)
        
        # Evaluation
        metrics = trainer.evaluate(
            test_loader, 
            device, 
            epoch=epoch, 
            total_epochs=args.epochs, 
            label_names=le.classes_
        )

        # Save Best Model with Timestamped Name
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_name = f"best_student_{run_id}.pth"
            torch.save({
                'run_id': run_id,
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'hyperparams': {'alpha': args.alpha, 'beta': args.beta, 'student': args.student_type},
                'loss': epoch_loss,
                'accuracy': metrics['accuracy'],
                'le': le
            }, os.path.join(SAVE_DIR, best_model_name))
            print(f"⭐ Epoch {epoch:02d}: New Best Model Saved ({epoch_loss:.4f})")

        # Log results to list
        epoch_data = {
            "run_id": run_id,
            "epoch": epoch,
            "loss": round(epoch_loss, 5),
            "accuracy": round(metrics['accuracy'], 5),
            "f1_score": round(metrics['f1'], 5),
            "precision": round(metrics['precision'], 5),
            "recall": round(metrics['recall'], 5)
        }
        results_history.append(epoch_data)

        # Save/Update CSV
        pd.DataFrame(results_history).to_csv(csv_path, index=False)
        
        # Clean console summary
        print(f"📊 Summary Ep {epoch:02d}: Loss={epoch_loss:.4f} | Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f}")

    print(f"\n[4/4] Complete! All artifacts saved in: {SAVE_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGKD for IoT Anomaly Detection")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default="./", help='Base directory for outputs')
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--student_type', choices=['tiny', 'small', 'medium'], default='small')
    parser.add_argument('--use_trust', action='store_true')
    
    args = parser.parse_args()
    main(args)