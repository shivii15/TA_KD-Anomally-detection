import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from data.data_loader import preprocess_iot_data

def train_anomaly_module(data_path, save_path="models/iso_forest_iot.pkl"):
    print("🔍 Loading data to train Anomaly Module...")
    # Load a representative sample (e.g., 5-10 parts) to learn 'benign' behavior
    X, y, le, _ = preprocess_iot_data(data_path, num_parts=10)
    
    # Identify the 'Benign' label index
    # Note: In CIC-IoT-2023, ensure you target the correct benign class string
    try:
        benign_idx = list(le.classes_).index('Benign')
    except ValueError:
        # Fallback if the label is different in your preprocessing
        benign_idx = 0 
        print(f"⚠️ 'Benign' label not found. Defaulting to index {benign_idx}")

    # Filter for Benign samples only
    X_benign = X[y == benign_idx]
    
    if len(X_benign) == 0:
        raise ValueError("❌ No benign samples found in the selected data parts!")

    print(f"🛠️ Training Isolation Forest on {len(X_benign)} benign samples...")
    # Parameters aligned with typical IoT anomaly detection
    iso_forest = IsolationForest(
        n_estimators=100, 
        contamination=0.01, # Assume 1% of 'benign' might be noise
        random_state=42,
        n_jobs=-1 # Use all CPU cores
    )
    
    iso_forest.fit(X_benign)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(iso_forest, save_path)
    print(f"✅ Anomaly Module saved to {save_path}")

if __name__ == "__main__":
    DATA_PATH = "/root/.cache/kagglehub/datasets/akashdogra/cic-iot-2023/versions/1"
    train_anomaly_module(DATA_PATH)