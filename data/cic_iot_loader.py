import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_cic_data(data_dir, num_parts=3):
    """
    SOTA-ready preprocessing for CIC-IoT-2023.
    Includes typo correction, stratified sampling, and artifact saving.
    """
    # 1. Find and Filter Files
    all_files = glob.glob(os.path.join(data_dir, "part-*.csv"))
    all_files.sort()
    
    # Handle -1 to load the full dataset for final PhD results
    selected_files = all_files if num_parts == -1 else all_files[:num_parts]
    
    if not selected_files:
        raise FileNotFoundError(f"❌ No CSV files found in {data_dir}")

    print(f"📂 Data Directory: {data_dir}")
    print(f"📦 Processing {len(selected_files)} files...")

    # Memory efficient reading
    dfs = []
    for f in selected_files:
        temp_df = pd.read_csv(f, low_memory=False)
        dfs.append(temp_df)
    
    df = pd.concat(dfs, ignore_index=True)

    # 2. Cleaning & Standardizing (Crucial for ResNet convergence)
    # Fix common typos in the CIC-IoT-2023 headers
    df.columns = [col.replace(' ', '_').replace('Magnitue', 'Magnitude') for col in df.columns]
    
    # Research Scope: Removing low-frequency web-based attacks to focus on Network/IoT layers
    to_remove = ['DictionaryBruteForce', 'BrowserHijacking', 'XSS', 
                 'Uploading_Attack', 'SqlInjection', 'CommandInjection', 'Backdoor_Malware']
    df = df[~df['label'].isin(to_remove)]

    # 3. Feature Extraction
    # Convert all to numeric, coercing errors to 0 (common in IoT sensor noise)
    X_raw = df.drop(columns=['label']).apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    le = LabelEncoder()
    y_data = le.fit_transform(df['label'].values)

    # 4. Feature Scaling
    # ResNet teachers are highly sensitive to unscaled inputs
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_raw)
    
    # Save objects for Inference and Student Synchronization
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    print(f"✅ Preprocessing complete. Feature count: {X_final.shape[1]}, Classes: {len(le.classes_)}")

    return X_final, y_data, le, scaler

def get_dataloaders(X, y, batch_size=1024):
    """
    Creates stratified loaders to maintain attack distribution balance.
    """
    # random_state=42 for reproducibility in your research paper
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.long)
    )
    
    # num_workers=4 can be added here if you are running on a multi-core Linux server
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader