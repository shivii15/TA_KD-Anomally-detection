import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_iot_data(data_path):
    """
    Loads CIC-IoT-2023 CSV, encodes string labels to integers, 
    scales features, and splits into train/test sets.
    """
    print(f"🔍 Reading dataset: {data_path}")
    df = pd.read_csv(data_path)
    
    # 1. Identify Label Column
    # In CIC-IoT-2023, the column is usually named 'label'
    if 'label' not in df.columns:
        raise ValueError(f"Label column not found. Available columns: {df.columns.tolist()[:5]}...")
    
    X = df.drop(columns=['label'])
    y = df['label']

    # 2. Convert Labels to Integers (Crucial Fix)
    # This turns 'DDoS-SynonymousIP_Flood' -> 0, 'Benign' -> 1, etc.
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"✅ Encoded {num_classes} distinct classes.")

    # 3. Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, scaler, le

def get_dataloaders(X_train, X_test, y_train, y_test, batch_size):
    """
    Converts numpy arrays to PyTorch Tensors and creates DataLoaders.
    """
    # Force conversion to numeric types to avoid 'object' or 'string' errors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Create Datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    # Create Loaders with GPU optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True if batch_size > 128 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True
    )

    return train_loader, test_loader