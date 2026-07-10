import os
import json
import argparse

import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="results"
    )
    parser.add_argument(
        "--output_dir",
        default="summary"
    )
    return parser.parse_args()

def load_experiment(folder):
    experiment_file = os.path.join(folder, "experiment.json")
    if os.path.exists(experiment_file):
        with open(experiment_file) as f:
            experiment = json.load(f)
    else:
        print(f"Skipping {folder} (no experiment.json)")
        return None

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
    
def main():
    args = parse_args()

    rows = []
    folders = sorted(
        [
            os.path.join(
                args.results_dir,
                d
            )
            for d in os.listdir(
                args.results_dir
            )
            if os.path.isdir(
                os.path.join(
                    args.results_dir,
                    d
                )
            )
        ]
    )
    print("\nFolders found:")

    for folder in folders:
        print(folder)
        try:
            rows.append(
                load_experiment(folder)
            )
        except Exception as e:
            print(f"\nError reading {folder}")
            print(e)
    df = pd.DataFrame(rows)
    print("\nRows collected:", len(rows))
    print(df)
    print(df.columns)
    order = [
        "Full TGKD",
        "Confidence",
        "Entropy",
        "Anomaly",
        "Disagreement",
        "Temperature",
        "Feature"
    ]

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
    print()
    print(df)
    print()
    print(
        f"Saved summary to {args.output_dir}"
    )

if __name__ == "__main__":
    main()