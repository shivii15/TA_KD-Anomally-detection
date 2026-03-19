import pandas as pd
import joblib
import os
import argparse
from sklearn.ensemble import IsolationForest
from data.data_loader import preprocess_iot_data

def train_anomaly_module(args):
    print(f"🔍 Loading data from: {args.data_path}")
    
    # We only need a small subset of the 169 files to learn "Benign" behavior
    X, y, le, _ = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    
    # Identify the 'Benign' label index dynamically
    try:
        benign_label = "Benign"
        if benign_label in le.classes_:
            benign_idx = list(le.classes_).index(benign_label)
        else:
            # Fallback if your LabelEncoder used different casing or names
            benign_idx = 0 
            print(f"⚠️ '{benign_label}' not found in classes {le.classes_}. Using index 0.")
    except Exception as e:
        benign_idx = 0
        print(f"⚠️ Error identifying benign label: {e}. Defaulting to index 0.")

    # Filter for Benign samples only
    X_benign = X[y == benign_idx]
    
    if len(X_benign) == 0:
        raise ValueError("❌ No benign samples found! Check your data path or num_parts.")

    print(f"🛠️ Training Isolation Forest on {len(X_benign)} benign samples...")
    
    # PDF Methodology: Isolation Forest for Anomaly Score A(x)
    iso_forest = IsolationForest(
        n_estimators=args.n_estimators, 
        contamination=args.contamination, 
        random_state=42,
        n_jobs=-1 
    )
    
    iso_forest.fit(X_benign)
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    joblib.dump(iso_forest, args.save_path)
    print(f"✅ Anomaly Module saved to: {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Anomaly Module for TGKD")
    
    # Data & Path Args
    parser.add_argument('--data_path', type=str, required=True, 
                        help="Path to CIC-IoT-2023 CSV files")
    parser.add_argument('--save_path', type=str, default="models/iso_forest_iot.pkl",
                        help="Where to save the trained .pkl model")
    parser.add_argument('--num_parts', type=int, default=10,
                        help="Number of data parts to use for training (default 10)")
    
    # Model Hyperparameters (Good for ablation studies in your thesis)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--contamination', type=float, default=0.01,
                        help="Expected proportion of outliers in the benign set")

    args = parser.parse_args()
    train_anomaly_module(args)