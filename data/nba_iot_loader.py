"""
nbaiot_loader.py
"""

import os
import glob
import pandas as pd
import kagglehub

from .common import (
    encode_labels,
    scale_features,
    save_preprocessing_objects
)


def load_nbaiot_dataset(
    data_dir=None,
    num_parts=-1
):

    print("\nLoading N-BaIoT dataset")

    if data_dir is None:

        data_dir = kagglehub.dataset_download(
            "mkashifn/nbaiot-dataset"
        )

    print(data_dir)

    all_files = glob.glob(
        os.path.join(data_dir, "*.csv")
    )

    ignore = {
        "features.csv",
        "device_info.csv",
        "data_summary.csv"
    }

    all_files = [
        file
        for file in all_files
        if os.path.basename(file) not in ignore
    ]

    all_files.sort()

    if num_parts == -1:
        selected_files = all_files
    else:
        selected_files = all_files[:num_parts]

    print(f"Loading {len(selected_files)} files")

    dfs = []

    for file in selected_files:

        filename = os.path.basename(file)

        print(filename)

        df = pd.read_csv(
            file,
            low_memory=False
        )

        parts = filename.replace(
            ".csv",
            ""
        ).split(".")

        label = "_".join(parts[1:])

        df["label"] = label

        dfs.append(df)

    df = pd.concat(
        dfs,
        ignore_index=True
    )

    X = (
        df
        .drop(columns=["label"])
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values
    )

    y, le = encode_labels(df["label"])

    X, scaler = scale_features(X)

    save_preprocessing_objects(
        scaler,
        le
    )

    print(f"Samples : {len(df)}")

    print(f"Features : {X.shape[1]}")

    print(f"Classes : {len(le.classes_)}")

    return X, y, le, scaler