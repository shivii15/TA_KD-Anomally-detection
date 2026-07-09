from .cic_iot_loader import load_cic_dataset
from .nba_iot_loader import load_nbaiot_dataset
from .common import get_dataloaders
import os
import kagglehub

def download_nbaiot():
    print("⬇️ Downloading N-BaIoT dataset...")
    path = kagglehub.dataset_download("mkashifn/nbaiot-dataset")
    return path

def download_cic():
    print("⬇️ Downloading CIC-IoT2023 dataset...")
    path = kagglehub.dataset_download("jaganadhg/cic-iot2023")
    return path

def resolve_dataset_path(dataset, data_path=None):
    """
    Resolve dataset location.

    Priority
    --------
    1. User supplied path
    2. Kaggle cache
    3. Automatic download
    """

    # ----------------------------------------
    # Case 1 : Server
    # ----------------------------------------
    if data_path is not None:

        if not os.path.exists(data_path):

            raise FileNotFoundError(
                f"Dataset directory not found:\n{data_path}"
            )

        print(f"📂 Using dataset from: {data_path}")

        return data_path

    # ----------------------------------------
    # Case 2 : Colab
    # ----------------------------------------

    print("📦 No dataset path supplied.")

    print("Checking local Kaggle cache...")

    if dataset == "nbaiot":

        return download_nbaiot()

    elif dataset == "cic":

        return download_cic()

    else:

        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

def load_dataset(args):
    dataset = args.dataset.lower()

    dataset_path = resolve_dataset_path(
        args.dataset,
        args.data_path
    )


    if dataset == "cic":

        return load_cic_dataset(
            data_dir=dataset_path,
            num_parts=args.num_parts
        )

    elif dataset == "nbaiot":

        return load_nbaiot_dataset(
            data_dir=dataset_path,
            num_parts=args.num_parts
        )

    else:

        raise ValueError(
            f"Unsupported dataset : {dataset}"
        )