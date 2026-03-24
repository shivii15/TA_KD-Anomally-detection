import pandas as pd
import numpy as np
import joblib
import os
import argparse
from sklearn.ensemble import IsolationForest
from data.data_loader import preprocess_iot_data

def train_anomaly_module(args):
    print(f"🔍 Loading data from: {args.data_path}")
    
    # 1. Load data using your specialized loader
    X, y, le, _ = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    
    # 2. Robust Label Detection (Handles 'Benign' or 'BenignTraffic')
    benign_idx = None
    possible_names = ["BenignTraffic", "Benign", "benign", "BENIGN"]
    
    for name in possible_names:
        if name in le.classes_:
            benign_idx = list(le.classes_).index(name)
            print(f"✅ Found benign label: '{name}' at index {benign_idx}")
            break

    if benign_idx is None:
        # Fallback: Many IoT datasets put Benign at index 0
        benign_idx = 0
        print(f"⚠️ Benign label not explicitly found. Defaulting to index 0: {le.classes_[0]}")

    # 3. Filter for Benign samples only (Crucial for One-Class Isolation Forest)
    X_benign = X[y == benign_idx]
    
    if len(X_benign) == 0:
        print(f"❌ Error: No samples found for index {benign_idx}. Classes are: {le.classes_}")
        return

    print(f"🛠️ Training Isolation Forest on {len(X_benign)} benign samples...")
    
    # 4. Initialize Isolation Forest (SOTA Configuration for IoT)
    # n_jobs=-1 ensures it uses all available server CPU cores
    iso_forest = IsolationForest(
        n_estimators=args.n_estimators, 
        contamination=args.contamination, 
        max_samples='auto',
        random_state=42,
        n_jobs=-1 
    )
    
    iso_forest.fit(X_benign)
    
    # 5. Save Model and Metadata
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # We save as a dictionary to keep the label information with the model
    model_package = {
        'model': iso_forest,
        'benign_label': le.classes_[benign_idx],
        'features_count': X.shape[1]
    }
    
    joblib.dump(model_package, args.save_path)
    print(f"✅ Anomaly Module + Metadata saved to: {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Anomaly Module for TGKD")
    
    parser.add_argument('--data_path', type=str, required=True, 
                        help="Path to CIC-IoT-2023 CSV files")
    parser.add_argument('--save_path', type=str, default="models/anomaly_detector_v1.pkl",
                        help="Where to save the trained .pkl model")
    parser.add_argument('--num_parts', type=int, default=10,
                        help="Number of data parts to use for training")
    
    # Hyperparameters for your Research Table
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--contamination', type=float, default=0.01)

    args = parser.parse_args()
    train_anomaly_module(args)