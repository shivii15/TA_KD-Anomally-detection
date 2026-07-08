"""
cic_iot_loader.py
"""

import os
import glob
import pandas as pd

from .common import (
    encode_labels,
    scale_features,
    save_preprocessing_objects
)


def load_cic_dataset(
    data_dir,
    num_parts=-1
):

    print(f"\nLoading CIC-IoT dataset")

    all_files = glob.glob(
        os.path.join(data_dir, "part-*.csv")
    )

    all_files.sort()

    if num_parts == -1:
        selected_files = all_files
    else:
        selected_files = all_files[:num_parts]

    print(f"Loading {len(selected_files)} files")

    dfs = []

    for file in selected_files:

        print(os.path.basename(file))

        df = pd.read_csv(
            file,
            low_memory=False
        )

        dfs.append(df)

    df = pd.concat(
        dfs,
        ignore_index=True
    )

    df.columns = (
        df.columns
        .str.replace(" ", "_")
        .str.replace("Magnitue", "Magnitude")
    )

    to_remove = [
        "DictionaryBruteForce",
        "BrowserHijacking",
        "XSS",
        "Uploading_Attack",
        "SqlInjection",
        "CommandInjection",
        "Backdoor_Malware"
    ]

    df = df[
        ~df["label"].isin(to_remove)
    ]

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