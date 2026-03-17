import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


def preprocess_iot_data(data_dir, num_parts=3):
    """
    Finds CSV parts, merges them, filters labels, and scales features.
    """
    # 1. Dynamically find all CSV parts
    all_files = glob.glob(os.path.join(data_dir, "part-*.csv"))
    all_files.sort()
    
    selected_files = all_files[:num_parts]
    if not selected_files:
        raise FileNotFoundError(f"No CSV parts found in {data_dir}")

    print(f"📦 Loading {len(selected_files)} parts for training...")
    
    dfs = []
    for f in selected_files:
        # Load with low_memory=False to handle the large IoT dataset types
        temp_df = pd.read_csv(f, low_memory=False)
        dfs.append(temp_df)
    
    df = pd.concat(dfs, ignore_index=True)

    # 2. Clean column names (fixes the Magnitude typo and spaces)
    df.columns = df.columns.str.replace(' ', '_').str.replace('Magnitue', 'Magnitude')

    # 3. Apply your Research Filter
    labels_to_remove = ['DictionaryBruteForce', 'BrowserHijacking', 'XSS', 
                        'Uploading_Attack', 'SqlInjection', 'CommandInjection', 'Backdoor_Malware']
    df = df[~df['label'].isin(labels_to_remove)]

    # 4. Separate Features and Labels
    X = df.drop(columns=['label']).apply(pd.to_numeric, errors='coerce').fillna(0).values
    y_raw = df['label'].values

    # 5. Label Encoding (String -> Int)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # 6. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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