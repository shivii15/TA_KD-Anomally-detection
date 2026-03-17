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
    
    # Logic for -1 (all files)
    selected_files = all_files if num_parts == -1 else all_files[:num_parts]
    print(f"📦 Processing {len(selected_files)} files...")

    dfs = [pd.read_csv(f, low_memory=False) for f in selected_files]
    df = pd.concat(dfs, ignore_index=True)

    # 2. Cleaning & Research Constraints
    df.columns = df.columns.str.replace(' ', '_').str.replace('Magnitue', 'Magnitude')
    
    # Specific filter for your project
    to_remove = ['DictionaryBruteForce', 'BrowserHijacking', 'XSS', 
                 'Uploading_Attack', 'SqlInjection', 'CommandInjection', 'Backdoor_Malware']
    df = df[~df['label'].isin(to_remove)]

    # 3. Features (X) and Labels (y)
    # Convert to numeric, handle NaNs, and get values immediately
    X_data = df.drop(columns=['label']).apply(pd.to_numeric, errors='coerce').fillna(0).values
    y_data = LabelEncoder().fit_transform(df['label'].values)

    # 4. Scaling (The line that crashed)
    scaler = StandardScaler()
    # FIX: We use X_data here, and return X_final
    X_final = scaler.fit_transform(X_data)
    
    # Save for Student use
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')

    return X_final, y_data, scaler

def get_dataloaders(X, y, batch_size=1024):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(val_ds, batch_size=batch_size, shuffle=False)