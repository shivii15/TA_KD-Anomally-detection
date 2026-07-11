import os
import json
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments_dir",
        default="experiments"
    )
    parser.add_argument(
        "--output_dir",
        default="summary"
    )
    return parser.parse_args()

def load_experiment(folder):
    experiment_file = os.path.join(
        folder,
        "experiment.json"
    )

    print("\nChecking:")
    print("Folder :", folder)
    print("Experiment file :", experiment_file)
    print("Exists :", os.path.exists(experiment_file))

    if not os.path.exists(experiment_file):
        print(f"Skipping {folder} (no experiment.json)")
        return None

    with open(experiment_file) as f:
        experiment = json.load(f)


    with open(
        os.path.join(folder, "metrics.json")
    ) as f:
        metrics = json.load(f)

    with open(
        os.path.join(
            folder,
            "analysis",
            "analysis_statistics.json"
        )
    ) as f:
        analysis = json.load(f)
        return {
            "Experiment":
                experiment["experiment"],
            "Dataset":
                experiment["dataset"],
            "Epochs":
                experiment["epochs"],
            "Accuracy":
                metrics["accuracy"],
            "Precision":
                metrics["precision"],
            "Recall":
                metrics["recall"],
            "F1":
                metrics["f1_score"],
            "Balanced Accuracy":
                metrics["balanced_accuracy"],
            "MCC":
                metrics["mcc"],
            "Mean Trust":
                analysis["trust"]["mean"],
            "Mean Confidence":
                analysis["confidence"]["mean"],
            "Mean Entropy":
                analysis["entropy"]["mean"],
            "Mean Anomaly":
                analysis["anomaly"]["mean"],
            "Mean Disagreement":
                analysis["disagreement"]["mean"],
            "Mean Temperature":
                analysis["temperature"]["mean"]
        }
    
def plot_metric(
    df,
    metric,
    output_dir,
    filename
):
    plt.figure(figsize=(8,5))
    bars = plt.bar(
        df["Experiment"],
        df[metric]
    )
    bars[0].set_hatch("//")

    plt.ylabel(metric)

    plt.title(metric)

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            filename
        ),
        dpi=300
    )

    plt.close()
    
def main():
    args = parse_args()

    rows = []

    folders = sorted(
        glob.glob(
            os.path.join(
                args.experiments_dir,
                "*",
                "results"
            )
        )
    )
    print("\nFolders found:")

    for folder in folders:
        print(folder)
        try:
            experiment = load_experiment(folder)
            if experiment is not None:
                rows.append(experiment)
        except Exception as e:
            print(f"\nError reading {folder}")
            print(e)
    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo valid experiments found.")
        return

    print("\nRows collected:", len(rows))
    print("\nLoaded Experiments")
    print(df[["Experiment","Accuracy","F1"]])
    print(df.columns)
    order = [
        "Full TGKD",
        "Ablation Confidence",
        "Ablation Entropy",
        "Ablation Anomaly",
        "Ablation Disagreement",
        "Ablation Temperature",
        "Ablation Feature"
    ]
    column_order = [
        "Experiment",
        "Dataset",
        "Epochs",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Balanced Accuracy",
        "MCC",
        "Mean Trust",
        "Mean Confidence",
        "Mean Entropy",
        "Mean Anomaly",
        "Mean Disagreement",
        "Mean Temperature"
    ]

    df = df[column_order]

    df = df.round(4)
    df["Experiment"] = pd.Categorical(
        df["Experiment"],
        categories=order,
        ordered=True
    )

    df = df.sort_values(
        "Experiment"
    )
    os.makedirs(
        args.output_dir,
        exist_ok=True
    )
    df.to_csv(
        os.path.join(
            args.output_dir,
            "experiments.csv"
        ),
        index=False
    )
    df.to_excel(
        os.path.join(
            args.output_dir,
            "experiments.xlsx"
        ),
        index=False
    )

    df.to_latex(
        os.path.join(
            args.output_dir,
            "ablation_table.tex"
        ),
        index=False,
        float_format="%.4f"
    )
    with open(
        os.path.join(
            args.output_dir,
            "ablation_table.md"
        ),
        "w"
    ) as f:

        f.write(
            df.to_markdown(index=False)
        )

    plot_metric(
        df,
        "Accuracy",
        args.output_dir,
        "accuracy_bar.png"
    )

    plot_metric(
        df,
        "Precision",
        args.output_dir,
        "precision_bar.png"
    )

    plot_metric(
        df,
        "Recall",
        args.output_dir,
        "recall_bar.png"
    )

    plot_metric(
        df,
        "F1",
        args.output_dir,
        "f1_bar.png"
    )

    plot_metric(
        df,
        "Balanced Accuracy",
        args.output_dir,
        "balanced_accuracy_bar.png"
    )

    plot_metric(
        df,
        "MCC",
        args.output_dir,
        "mcc_bar.png"
    )
    print()
    print(df)
    print()
    print(
        f"Saved summary to {args.output_dir}"
    )

if __name__ == "__main__":
    main()