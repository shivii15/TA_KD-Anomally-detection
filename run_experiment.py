"""
Master Runner for TGKD Framework

Runs the complete pipeline:

1. Train Teachers
2. Train Isolation Forest
3. Train TGKD Student
4. Evaluate
5. SNAP Analysis
6. Feature Analysis
7. ROC/PR Analysis
8. Performance Analysis
"""

import argparse
import subprocess
import os
import glob


def run(cmd):
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 80)

    subprocess.run(cmd, check=True)


def newest_file(pattern):
    files = glob.glob(pattern)

    if len(files) == 0:
        raise FileNotFoundError(pattern)

    files.sort(key=os.path.getmtime)

    return files[-1]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="nbaiot")

    parser.add_argument("--data_path", default=None)

    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--batch_size", type=int, default=1024)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--skip_teacher", action="store_true")

    parser.add_argument("--skip_iso", action="store_true")

    parser.add_argument("--skip_student", action="store_true")

    parser.add_argument("--skip_test", action="store_true")

    parser.add_argument("--skip_analysis", action="store_true")

    args = parser.parse_args()

    # --------------------------------------------------
    # Train Teachers
    # --------------------------------------------------

    teachers = [
        "dnn",
        "resnet",
        "transformer",
        "lstm"
    ]

    if not args.skip_teacher:

        for teacher in teachers:

            cmd = [
                "python",
                "train_teacher.py",
                "--teacher",
                teacher,
                "--dataset",
                args.dataset,
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size)
            ]

            if args.data_path is not None:

                cmd += [
                    "--data_path",
                    args.data_path
                ]

            run(cmd)

    # --------------------------------------------------
    # Train Isolation Forest
    # --------------------------------------------------

    if not args.skip_iso:

        cmd = [
            "python",
            "train_anomaly_detector.py",
            "--dataset",
            args.dataset
        ]

        if args.data_path is not None:

            cmd += [
                "--data_path",
                args.data_path
            ]

        run(cmd)

    # --------------------------------------------------
    # Locate checkpoints
    # --------------------------------------------------

    dnn = newest_file(
        f"models/Teacher_dnn_{args.dataset}_*.pth"
    )

    resnet = newest_file(
        f"models/Teacher_resnet_{args.dataset}_*.pth"
    )

    transformer = newest_file(
        f"models/Teacher_transformer_{args.dataset}_*.pth"
    )

    lstm = newest_file(
        f"models/Teacher_lstm_{args.dataset}_*.pth"
    )

    iso = newest_file(
        f"models/IsolationForest_{args.dataset}_*.pkl"
    )

    # --------------------------------------------------
    # Train Student
    # --------------------------------------------------

    if not args.skip_student:

        cmd = [
            "python",
            "main.py",

            "--dataset",
            args.dataset,

            "--epochs",
            str(args.epochs),

            "--batch_size",
            str(args.batch_size),

            "--lr",
            str(args.lr),

            "--dnn_path",
            dnn,

            "--resnet_path",
            resnet,

            "--trans_path",
            transformer,

            "--lstm_path",
            lstm,

            "--iso_path",
            iso
        ]

        if args.data_path is not None:

            cmd += [
                "--data_path",
                args.data_path
            ]

        run(cmd)

    student = newest_file(
        "models/Student_MultiTeacher*.pth"
    )

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

    if not args.skip_test:

        cmd = [
            "python",
            "test.py",

            "--dataset",
            args.dataset,

            "--checkpoint",
            student,

            "--iso_path",
            iso,

            "--resnet_path",
            resnet,

            "--trans_path",
            transformer,

            "--lstm_path",
            lstm
        ]

        if args.data_path is not None:

            cmd += [
                "--data_path",
                args.data_path
            ]

        run(cmd)

    # --------------------------------------------------
    # Locate latest results
    # --------------------------------------------------

    results = glob.glob(
        f"results/student_{args.dataset}_*"
    )

    results.sort(key=os.path.getmtime)

    result_dir = results[-1]

    # --------------------------------------------------
    # Analysis
    # --------------------------------------------------

    if not args.skip_analysis:

        analyses = [

            "analysis/snap_analysis.py",

            "analysis/feature_analysis.py",

            "analysis/roc_pr_analysis.py"

        ]

        for script in analyses:

            run([
                "python",
                script,
                "--result_dir",
                result_dir
            ])

    print("\n")
    print("=" * 80)
    print("TGKD PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("Latest Student :", student)
    print("Results Folder :", result_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()