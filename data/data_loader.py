import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_iot_data(data_path):
    df = pd.read_csv(data_path)
    
    # 1. Separate features and labels
    X = df.drop(columns=['label'])
    y = df['label']

    # 2. Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. Create X_benign (CRITICAL FOR TRUST GATE)
    # Find the integer index for 'Benign' (it might be 0, 1, etc.)
    benign_idx = np.where(le.classes_ == 'Benign')[0][0]
    # Filter the raw features for only benign traffic
    X_benign_raw = X[y == 'Benign'] 

    # 4. Scale everything
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Scale the benign subset separately using the same scaler
    X_benign_scaled = scaler.transform(X_benign_raw)

    # 5. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # RETURN THE ARRAY, NOT THE SCALER OBJECT
    return X_train, X_test, y_train, y_test, X_benign_scaled, le

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
        num_workers=2, 
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