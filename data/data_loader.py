import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import glob
import joblib # To save the scaler for the student

def preprocess_iot_data(data_dir, num_parts):
    # 1. Find all parts
    all_files = glob.glob(os.path.join(data_dir, "part-*.csv"))
    all_files.sort()
    
    # 2. Slice based on argument
    if num_parts == -1:
        selected_files = all_files
        print(f"🚀 Training on FULL dataset ({len(selected_files)} files)!")
    else:
        selected_files = all_files[:num_parts]
        print(f"📦 Training on {len(selected_files)} parts.")

    # ... (Loading and Filtering logic as before) ...
    
    # 3. Save Scaler (Crucial!)
    # The student MUST use the same scaling parameters as the teacher.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.pkl') 
    
    return X_scaled, y, le


def get_dataloaders(X, y, batch_size=1024, test_size=0.2):
    """
    Splits data and returns two separate DataLoaders.
    """
    # 1. Split into Train and Validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # 2. Convert to Tensors
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.long)
    )

    # 3. Create Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader  # This now returns TWO values