"""
common.py

Common preprocessing utilities used by all datasets.
"""

import os
import joblib
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# -------------------------------------------------------
# Label Encoding
# -------------------------------------------------------

def encode_labels(labels):
    """
    Encodes string labels into integers.
    """

    le = LabelEncoder()

    y = le.fit_transform(labels)

    return y, le


# -------------------------------------------------------
# Feature Scaling
# -------------------------------------------------------

def scale_features(X):
    """
    Standardize features.
    """

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    return X, scaler


# -------------------------------------------------------
# Save preprocessing objects
# -------------------------------------------------------

def save_preprocessing_objects(
    scaler,
    label_encoder,
    save_dir="models"
):

    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(
        scaler,
        os.path.join(save_dir, "scaler.pkl")
    )

    joblib.dump(
        label_encoder,
        os.path.join(save_dir, "label_encoder.pkl")
    )


# -------------------------------------------------------
# Create PyTorch DataLoaders
# -------------------------------------------------------

# -------------------------------------------------------
# Create PyTorch DataLoaders
# -------------------------------------------------------

def get_dataloaders(
    X,
    y,
    batch_size=1024,
    train_size=0.80,
    val_size=0.10,
    test_size=0.10,
    random_state=42
):
    """
    Split dataset into

        Train : 80%
        Validation : 10%
        Test : 10%

    Returns
    -------
    train_loader
    val_loader
    test_loader
    """

    # --------------------------------------------------
    # First Split
    # Train (80%) + Temp (20%)
    # --------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1 - train_size),
        stratify=y,
        random_state=random_state
    )

    # --------------------------------------------------
    # Second Split
    # Temp -> Validation (10%) + Test (10%)
    # --------------------------------------------------

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state
    )

    # --------------------------------------------------
    # PyTorch Datasets
    # --------------------------------------------------

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    print("\nDataset Split")
    print("-" * 40)
    print(f"Train Samples      : {len(train_ds):,}")
    print(f"Validation Samples : {len(val_ds):,}")
    print(f"Test Samples       : {len(test_ds):,}")
    print("-" * 40)

    return train_loader, val_loader, test_loader