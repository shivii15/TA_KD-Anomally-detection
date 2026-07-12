import subprocess
import argparse
import glob
import os
import json

def parse_args():

    parser = argparse.ArgumentParser(
        description="TGKD Ablation Experiment Manager"
    )

    parser.add_argument(
        "--dataset",
        choices=["nbaiot", "cic"],
        default="nbaiot"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--num_parts",
        type=int,
        default=None,
        help="Number of dataset partitions to load."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--feat_weight",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--temp_base",
        type=float,
        default=3.0
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Dataset directory."
    )

    parser.add_argument(
        "--resnet_path",
        type=str,
        default=None,
        help="Path to ResNet teacher."
    )

    parser.add_argument(
        "--trans_path",
        type=str,
        default=None,
        help="Path to Transformer teacher."
    )

    parser.add_argument(
        "--lstm_path",
        type=str,
        default=None,
        help="Path to LSTM teacher."
    )

    parser.add_argument(
        "--iso_path",
        type=str,
        default=None,
        help="Path to Isolation Forest."
    )
    return parser.parse_args()

def latest(pattern):
    files = glob.glob(pattern)

    if len(files) == 0:
        raise FileNotFoundError(
            f"No checkpoint found matching:\n{pattern}"
        )

    files.sort(key=os.path.getmtime)

    return files[-1]

def main():
    args = parse_args()

    DATASET = args.dataset
    EPOCHS = args.epochs
    BATCH = args.batch_size
    NUM_PARTS = args.num_parts
    LR = args.lr
    FEAT_WEIGHT = args.feat_weight
    TEMP_BASE = args.temp_base
    print("\nExperiment Manager")
    print("="*70)

    print("Dataset :", DATASET)
    print("Epochs :", EPOCHS)
    print("Batch :", BATCH)
    print("Num Parts :", NUM_PARTS)
    print("Learning Rate :", LR)
    print("Feature Weight :", FEAT_WEIGHT)
    print("Temperature :", TEMP_BASE)

    print("="*70)
    ABLATIONS = [

        ("Full TGKD",
        "Student_TGKD",
        []),

        ("Ablation Confidence",
        "Student_TGKD_AblationConfidence",
        ["--disable_confidence"]),

        ("Ablation Entropy",
        "Student_TGKD_AblationEntropy",
        ["--disable_entropy"]),

        ("Ablation Anomaly",
        "Student_TGKD_AblationAnomaly",
        ["--disable_anomaly"]),

        ("Ablation Disagreement",
        "Student_TGKD_AblationDisagreement",
        ["--disable_disagreement"]),

        ("Ablation Temperature",
        "Student_TGKD_AblationTemperature",
        ["--disable_temperature"]),

        ("Ablation Feature",
        "Student_TGKD_AblationFeature",
        ["--disable_feature"])
    ]



    resnet = (
        args.resnet_path
        if args.resnet_path
        else latest(f"models/Teacher_resnet_{DATASET}_*.pth")
    )

    transformer = (
        args.trans_path
        if args.trans_path
        else latest(f"models/Teacher_transformer_{DATASET}_*.pth")
    )

    lstm = (
        args.lstm_path
        if args.lstm_path
        else latest(f"models/Teacher_lstm_{DATASET}_*.pth")
    )

    iso = (
        args.iso_path
        if args.iso_path
        else latest(f"models/IsolationForest_{DATASET}_*.pkl")
    )

    print("\nTeacher Models")
    print("-"*50)
    print(resnet)
    print(transformer)
    print(lstm)
    print(iso)
    print("-"*50)

    for name, save_name, flags in ABLATIONS:

        print("\n")
        print("="*80)
        print(name)
        print("="*80)

        cmd = [
            "python",
            "main.py",
            "--dataset", DATASET,
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH),
            "--save_name", save_name,
            "--resnet_path", resnet,
            "--trans_path", transformer,
            "--lstm_path", lstm,
            "--iso_path", iso
        ]

        cmd.extend(flags)

        if args.data_path is not None:
            cmd.extend([
                "--data_path",
                args.data_path
            ])

        if NUM_PARTS is not None:

            cmd.extend([
                "--num_parts",
                str(NUM_PARTS)
            ])


        try:
            subprocess.run(
                cmd,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"\nExperiment failed: {name}")
            print(e)
            continue


        # ----------------------------------------------------
        # Locate newly created checkpoint
        # ----------------------------------------------------

        experiment_dirs = glob.glob(
            os.path.join(
                "experiments",
                f"{save_name}_*"
            )
        )

        if len(experiment_dirs) == 0:
            raise RuntimeError(
                f"No experiment folder found for {save_name}"
            )

        experiment_dirs.sort(
            key=os.path.getmtime
        )

        experiment_dir = experiment_dirs[-1]

        checkpoint = os.path.join(
            experiment_dir,
            "checkpoints",
            "best.pth"
        )

        if not os.path.exists(checkpoint):

            raise RuntimeError(
                f"Checkpoint not found:\n{checkpoint}"
            )


        print("\nLatest Checkpoint")

        print(checkpoint)

        test_cmd = [
            "python",
            "test.py",
            "--dataset", DATASET,
            "--model","student",
            "--checkpoint", checkpoint,
            "--resnet_path", resnet,
            "--trans_path", transformer,
            "--lstm_path", lstm,
            "--iso_path", iso
        ]
        if NUM_PARTS is not None:

            test_cmd.extend([
                "--num_parts",
                str(NUM_PARTS)
            ])
        if args.data_path is not None:
            test_cmd.extend([
                "--data_path",
                args.data_path
            ])

        print("\nRunning Test Command:")
        print(" ".join(test_cmd))


        try:
            subprocess.run(
                test_cmd,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"\nExperiment failed: {name}")
            print(e)
            continue


        result_dir = os.path.join(
            "experiments",
            save_name,
            "results"
        )

        if not os.path.exists(result_dir):
            raise RuntimeError(
                f"Result directory not found:\n{result_dir}"
            )


        # ----------------------------------------------------
        # Save Experiment Metadata
        # ----------------------------------------------------

        experiment_info = {
            "experiment": name,
            "save_name": save_name,
            "dataset": DATASET,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "learning_rate": LR,
            "feature_weight": FEAT_WEIGHT,
            "base_temperature": TEMP_BASE,
            "num_parts": NUM_PARTS,
            "checkpoint": checkpoint,

            # ADD THIS
            #"experiment_directory": experiment_dir,

            "teacher_committee": [
                os.path.basename(resnet),
                os.path.basename(transformer),
                os.path.basename(lstm)
            ],
            "isolation_forest": os.path.basename(iso),
            "ablation_flags": flags
        }

        with open(
            os.path.join(
                result_dir,
                "experiment.json"
            ),
            "w"
        ) as f:

            json.dump(
                experiment_info,
                f,
                indent=4
            )

        print("✓ Experiment metadata saved.")

        print(result_dir)

        #----------------
        # SNAP Analysis 
        #----------------

        snap = [
            "python",
            "analysis/snap_analysis.py",
            "--result_dir",
            result_dir
        ]


        try:
            subprocess.run(
                snap,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"\nExperiment failed: {name}")
            print(e)
            continue


        #------------------
        # Feature Analysis 
        #------------------

        feature = [

            "python",

            "analysis/feature_analysis.py",

            "--result_dir",

            result_dir

        ]


        try:
            subprocess.run(
                feature,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"\nExperiment failed: {name}")
            print(e)
            continue


        #------------------
        # ROC Analysis 
        #------------------

        roc = [

            "python",

            "analysis/roc_pr_analysis.py",

            "--result_dir",

            result_dir

        ]


        try:
            subprocess.run(
                roc,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"\nExperiment failed: {name}")
            print(e)
            continue

    print("\nAll ablations completed.")

    

if __name__ == "__main__":
    main()