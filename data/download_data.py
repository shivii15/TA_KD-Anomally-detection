import os
import argparse
import kagglehub


# -------------------------------------------------------
# Dataset Registry
# -------------------------------------------------------

DATASETS = {
    "nbaiot": "mkashifn/nbaiot-dataset",
    "cic": "akashdogra/cic-iot-2023"
}


# -------------------------------------------------------
# Download Function
# -------------------------------------------------------

def download_dataset(dataset_name):

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    print(f"\n📡 Downloading {dataset_name.upper()} dataset...")

    dataset_path = kagglehub.dataset_download(
        DATASETS[dataset_name]
    )

    print(f"✅ Download complete.")

    print(f"📂 Location : {dataset_path}")

    return dataset_path


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Download datasets using KaggleHub."
    )

    parser.add_argument(
        "--dataset",
        choices=["nbaiot", "cic", "all"],
        default="all",
        help="Dataset to download."
    )

    args = parser.parse_args()

    # ----------------------------------------
    # Project root
    # ----------------------------------------

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    PATH_FILE = os.path.join(
        BASE_DIR,
        ".data_path"
    )

    downloaded_paths = {}

    # ----------------------------------------
    # Download
    # ----------------------------------------

    if args.dataset == "all":

        datasets = ["nbaiot", "cic"]

    else:

        datasets = [args.dataset]

    for dataset in datasets:

        path = download_dataset(dataset)

        downloaded_paths[dataset] = path

    # ----------------------------------------
    # Save paths
    # ----------------------------------------

    with open(PATH_FILE, "w") as f:

        for dataset, path in downloaded_paths.items():

            f.write(f"{dataset}={path}\n")

    print("\n===========================================")
    print("All requested datasets downloaded.")
    print(f"Dataset paths saved in: {PATH_FILE}")
    print("===========================================")


if __name__ == "__main__":
    main()