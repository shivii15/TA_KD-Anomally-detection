import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

def preprocess_iot_data(file_path):
    df = pd.read_csv(file_path)
    
    # Feature Selection: The 47 core features for CIC-IoT-2023
    # Note: Replace with actual column names from your dataset
    X = df.drop(columns=['label']) 
    y = df['label'].values
    
    # Robust scaling for network traffic (handles outliers/spikes better)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Extract "Golden Set" (Benign only) for the Isolation Forest
    X_benign = X_train[y_train == 0] # Assuming 0 is 'Benign'
    
    return X_train, X_test, y_train, y_test, X_benign

def get_dataloaders(X_train, X_test, y_train, y_test, batch_size=1024):
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader