import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_paths):
    # CIC-IoT-2023 has 47 specific features + 1 label
    # This script would iterate through part-000x.csv files
    # 1. Drop NaN/Infinity values
    # 2. Map 34 attack strings to integers [0-33]
    # 3. Scale features using StandardScaler
    pass

def save_splits(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Save as .pt (PyTorch tensors) for faster loading in train_distill.py
    torch.save(X_train, 'data/train_features.pt')
    ...