import subprocess

import glob
import os

def latest(pattern):
    files = glob.glob(pattern)

    if len(files) == 0:
        raise FileNotFoundError(
            f"No checkpoint found matching:\n{pattern}"
        )

    files.sort(key=os.path.getmtime)

    return files[-1]

def main():
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

    DATASET = "nbaiot"

    EPOCHS = 100

    BATCH = 1024
    resnet = latest(
        f"models/Teacher_resnet_{DATASET}_*.pth"
    )

    transformer = latest(
        f"models/Teacher_transformer_{DATASET}_*.pth"
    )

    lstm = latest(
        f"models/Teacher_lstm_{DATASET}_*.pth"
    )

    iso = latest(
        f"models/IsolationForest_{DATASET}_*.pkl"
    )

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

        subprocess.run(
            cmd,
            check=True
        )

        # ----------------------------------------------------
        # Locate newly created checkpoint
        # ----------------------------------------------------

        checkpoints = glob.glob(
            f"models/{save_name}_*_best.pth"
        )

        checkpoints.sort(
            key=os.path.getmtime
        )

        checkpoint = checkpoints[-1]

        print("\nLatest Checkpoint")

        print(checkpoint)

        test_cmd = [

            "python",

            "test.py",

            "--dataset", DATASET,

            "--checkpoint", checkpoint,

            "--resnet_path", resnet,

            "--trans_path", transformer,

            "--lstm_path", lstm,

            "--dnn_path", dnn,

            "--iso_path", iso
        ]

        subprocess.run(
            test_cmd,
            check=True
        )

        results = glob.glob(
            f"results/student_{DATASET}_*"
        )

        results.sort(
            key=os.path.getmtime
        )

        result_dir = results[-1]

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

        subprocess.run(
            snap,
            check=True
        )

        #------------------
        # Feature Analysis 
        #------------------

        feature = [

            "python",

            "analysis/feature_analysis.py",

            "--result_dir",

            result_dir

        ]

        subprocess.run(
            feature,
            check=True
        )

        #------------------
        # ROC Analysis 
        #------------------

        roc = [

            "python",

            "analysis/roc_pr_analysis.py",

            "--result_dir",

            result_dir

        ]

        subprocess.run(
            roc,
            check=True
        )



    print("\nAll ablations completed.")

    

if __name__ == "__main__":
    main()