import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

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

    le = LabelEncoder()
    y = le.fit_transform(df['label']) # Assuming 'label' is your column name
    
    # Extract "Golden Set" (Benign only) for the Isolation Forest
    #X_benign = X_train[y_train == 0] # Assuming 0 is 'Benign'
    # 1. Identify the label column name (usually 'label')
    label_col = 'label' 

    # 2. Separate Benign data
    X_benign_df = df[df[label_col] == 'BenignTraffic']

    # 3. CRITICAL: Drop the label column so only numeric features remain
    X_benign = X_benign_df.drop(columns=[label_col])

    # 4. Optional: Ensure all other columns are numeric
    X_benign = X_benign.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 2. Add a check to prevent the crash
    if len(X_benign) == 0:
        print("Warning: No benign samples found! Using a small subset of training data instead.")
        X_benign = X_train[:1000] # Fallback so the code doesn't crash
    
    return X_train, X_test, y_train, y_test, X_benign, le

def get_dataloaders(X_train, X_test, y_train, y_test, batch_size=1024):
    # 1. Initialize the encoder
    le = LabelEncoder()
    
    # 2. Convert labels to integers (e.g., "DDoS" -> 0, "Benign" -> 1)
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    # 3. Now the tensors will work perfectly
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    #train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    #test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


class IoTRobustDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths # List of part*.csv files

    def __getitem__(self, idx):
        # Load only the specific row needed for this index
        # Or better: load one part file at a time and cache it
        ...