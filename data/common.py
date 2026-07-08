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

def get_dataloaders(
    X,
    y,
    batch_size=1024,
    test_size=0.2,
    random_state=42
):

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

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

    return train_loader, val_loader