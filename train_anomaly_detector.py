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
        # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    os.makedirs("models", exist_ok=True)

    save_path = os.path.join(
        "models",
        f"{args.save_name}_{args.dataset}_{timestamp}.pkl"
    )
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
    # --------------------------------------------------
    # Automatically detect the benign class
    # --------------------------------------------------

    benign_idx = None

    for idx, label in enumerate(le.classes_):

        if "benign" in label.lower():

            benign_idx = idx

            print(f"✅ Benign Class Detected : {label}")

            break

    if benign_idx is None:

        raise ValueError(
            f"Could not find a benign class.\nAvailable classes : {list(le.classes_)}"
        )
    

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
    
    
    # We include the scaler so the Trust-Gate can normalize raw inputs in real-time
    #model_package = {
    #    'model': iso_forest,
    #    'scaler': scaler,
    #    'benign_label': le.classes_[benign_idx],
    #    'features_count': X.shape[1],
    #    'train_samples': len(X_benign),
    #    'timestamp': timestamp
    #}

    model_package = {

        # --------------------------------------------------
        # Model
        # --------------------------------------------------
        "model": iso_forest,

        # --------------------------------------------------
        # Dataset Information
        # --------------------------------------------------
        "dataset": args.dataset,
        "timestamp": timestamp,

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------
        "scaler": scaler,
        "label_encoder": le,

        # --------------------------------------------------
        # Dataset Statistics
        # --------------------------------------------------
        "feature_dim": X.shape[1],
        "num_classes": len(le.classes_),
        "train_samples": len(X_benign),

        # --------------------------------------------------
        # Labels
        # --------------------------------------------------
        "classes": list(le.classes_),
        "benign_label": le.classes_[benign_idx],

        # --------------------------------------------------
        # Isolation Forest Parameters
        # --------------------------------------------------
        "n_estimators": args.n_estimators,
        "contamination": args.contamination

    }
    
    joblib.dump(model_package, save_path)
        
    print("\n" + "-" * 50)

    print("✅ Isolation Forest Trained Successfully")

    print("-" * 50)

    print(f"Dataset        : {args.dataset}")
    print(f"Samples        : {len(X_benign):,}")
    print(f"Feature Dim    : {X.shape[1]}")
    print(f"Benign Class   : {le.classes_[benign_idx]}")
    print(f"Trees          : {args.n_estimators}")
    print(f"Contamination  : {args.contamination}")

    print(f"\nSaved Model    : {save_path}")

    print("-" * 50)

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
    parser.add_argument(
        "--save_name",
        type=str,
        default="IsolationForest",
        help="Base name for the saved anomaly detector"
    )
    parser.add_argument('--num_parts', type=int, default=10)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--contamination', type=float, default=0.01)

    args = parser.parse_args()

    if args.dataset == "cic" and args.data_path is None:
        raise ValueError(
            "--data_path is required when using the CIC-IoT dataset."
        )
    train_anomaly_module(args)