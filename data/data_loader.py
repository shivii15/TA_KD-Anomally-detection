import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def preprocess_iot_data(data_dir, num_parts=3):
    # 1. Find and Filter Files
    all_files = glob.glob(os.path.join(data_dir, "part-*.csv"))
    all_files.sort()
    
    # Handle the "All Parts" argument
    selected_files = all_files if num_parts == -1 else all_files[:num_parts]
    print(f"📂 Loading data from: {data_dir}")
    print(f"📦 Processing {len(selected_files)} files...")

    dfs = [pd.read_csv(f, low_memory=False) for f in selected_files]
    df = pd.concat(dfs, ignore_index=True)

    # 2. Cleaning & Fixing Typos
    df.columns = df.columns.str.replace(' ', '_').str.replace('Magnitue', 'Magnitude')
    
    # Research Constraints: Filter attacks
    to_remove = ['DictionaryBruteForce', 'BrowserHijacking', 'XSS', 
                 'Uploading_Attack', 'SqlInjection', 'CommandInjection', 'Backdoor_Malware']
    df = df[~df['label'].isin(to_remove)]

    # 3. Separate Features and Labels
    # Use 'X_raw' to avoid the NameError
    X_raw = df.drop(columns=['label']).apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    le = LabelEncoder()
    y_data = le.fit_transform(df['label'].values)

    # 4. Scaling (Fixed the variable name here)
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_raw)
    
    # Save objects for Student synchronization
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')

    return X_final, y_data, le, scaler

def get_dataloaders(X, y, batch_size=1024):
    # Stratified split ensures rare attacks are in both sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(val_ds, batch_size=batch_size, shuffle=False)