import pandas as pd
import numpy as np
import joblib
import os
import argparse
from datetime import datetime
from sklearn.ensemble import IsolationForest
from data.data_loader import load_dataset

def train_anomaly_module(args):
    timestamp = datetime.now().strftime("%Y%m%d")
    print(f"🚀 Starting Anomaly Module Training | {timestamp}")
    
    # 1. Load data using your specialized loader
    # We capture 'scaler' to ensure the detector uses the same normalization as teachers
    X, y, le, scaler = load_dataset(args)

    print(f"\nDataset : {args.dataset}")

    print("\nDataset Summary")
    print("-" * 40)
    print(f"Samples      : {len(X):,}")
    print(f"Features     : {X.shape[1]}")
    print(f"Classes      : {len(le.classes_)}")
    print("-" * 40)
    
    # 2. Robust Label Detection (Handles 'Benign' variants)
    benign_idx = None
    possible_names = ["BenignTraffic", "Benign", "benign", "BENIGN"]
    
    for name in possible_names:
        if name in le.classes_:
            benign_idx = list(le.classes_).index(name)
            print(f"✅ Target Label Found: '{name}' (Index: {benign_idx})")
            break

    if benign_idx is None:
        benign_idx = 0
        print(f"⚠️ Warning: Benign label not found. Defaulting to index 0: {le.classes_[0]}")

    # 3. Filter for Benign samples (Semi-Supervised Approach)
    X_benign = X[y == benign_idx]
    
    if len(X_benign) == 0:
        print(f"❌ Error: Zero benign samples found. Available: {le.classes_}")
        return

    print(f"🛠️ Training Isolation Forest on {len(X_benign)} clean samples...")
    
    # 4. SOTA Configuration for IoT Anomaly Detection
    # Using 'auto' for max_samples ensures the model stays efficient on large datasets
    iso_forest = IsolationForest(
        n_estimators=args.n_estimators, 
        contamination=args.contamination, 
        max_samples='auto',
        random_state=42,
        n_jobs=-1 # Parallel processing for speed
    )
    
    iso_forest.fit(X_benign)
    
    # 5. Save Model + Essential Metadata for Distillation
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # We include the scaler so the Trust-Gate can normalize raw inputs in real-time
    model_package = {
        'model': iso_forest,
        'scaler': scaler,
        'benign_label': le.classes_[benign_idx],
        'features_count': X.shape[1],
        'train_samples': len(X_benign),
        'timestamp': timestamp
    }
    
    joblib.dump(model_package, args.save_path)
    
    print("-" * 30)
    print(f"✅ SUCCESS: Anomaly Module Ready")
    print(f"📂 Saved to: {args.save_path}")
    print(f"📏 Feature Dimension: {X.shape[1]}")
    print(f"🌲 Trees: {args.n_estimators} | Contamination: {args.contamination}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOTA Anomaly Module for Multi-Teacher TGKD")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cic",
        choices=["cic", "nbaiot"],
        help="Dataset to use"
    )
    #parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Dataset directory (optional for N-BaIoT)"
    )
    parser.add_argument('--save_path', type=str, default="models/iso_forest_iot_resnet_24march.pkl")
    parser.add_argument('--num_parts', type=int, default=10)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--contamination', type=float, default=0.01)

    args = parser.parse_args()

    if args.dataset == "cic" and args.data_path is None:
        raise ValueError(
            "--data_path is required when using the CIC-IoT dataset."
        )
    train_anomaly_module(args)