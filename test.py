import argparse
import os
import json
from datetime import datetime

import torch
import torch.nn as nn

from data.data_loader import load_dataset, get_dataloaders
from config.default_config import DEFAULT_CONFIG

from models.model import (
    TeacherDNN,
    TeacherResNet,
    TeacherTransformer,
    TeacherLSTM,
    StudentMLP
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)



def parse_args():

    parser = argparse.ArgumentParser(
        description="TGKD Evaluation Pipeline"
    )

    parser.add_argument(
        "--dataset",
        choices=["cic", "nbaiot"],
        required=True
    )

    parser.add_argument(
        "--data_path",
        default=None,
        type=str
    )

    parser.add_argument(
        "--model",
        choices=[
            "student",
            "dnn",
            "resnet",
            "transformer",
            "lstm"
        ],
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str
    )

    parser.add_argument(
        "--batch_size",
        default=DEFAULT_CONFIG["batch_size"],
        type=int
    )

    parser.add_argument(
        "--num_parts",
        default=DEFAULT_CONFIG["num_parts"],
        type=int
    )

    return parser.parse_args()

def main():

    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\nDevice : {device}")
    print(f"Dataset : {args.dataset}")
    print(f"Model : {args.model}")
    print(f"Checkpoint : {args.checkpoint}")

    X, y, le, scaler = load_dataset(args)
    _,_, test_loader = get_dataloaders(
        X,
        y,
        batch_size=args.batch_size
    )

    input_dim = X.shape[1]
    num_classes = len(le.classes_)

    print("\nDataset Summary")
    print("-" * 40)
    print(f"Samples : {len(X):,}")
    print(f"Features : {input_dim}")
    print(f"Classes : {num_classes}")
    print("-" * 40)

    TEACHER_MODELS = {
        "dnn": TeacherDNN,
        "resnet": TeacherResNet,
        "transformer": TeacherTransformer,
        "lstm": TeacherLSTM,
    }

    STUDENT_MODELS = {
        "student": StudentMLP,
    }

    model_map = {
        **TEACHER_MODELS,
        **STUDENT_MODELS,
    }

    model = model_map[args.model](
        input_dim,
        num_classes
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )
    print("\nCheckpoint Contents")
    print("-" * 40)

    for key in checkpoint.keys():
        print(key)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_logits = []
    all_features = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits, features = model(batch_x)
            probabilities = torch.softmax(
                logits,
                dim=1
            )
            predictions = torch.argmax(
                probabilities,
                dim=1
            )
            all_labels.append(batch_y.cpu())
            all_predictions.append(predictions.cpu())
            all_probabilities.append(probabilities.cpu())
            all_logits.append(logits.cpu())
            all_features.append(features.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_predictions = torch.cat(all_predictions).numpy()
    all_probabilities = torch.cat(all_probabilities).numpy()
    all_logits = torch.cat(all_logits).numpy()
    all_features = torch.cat(all_features).numpy()

    # -------------------------------------------------
    # Sanity Checks
    # -------------------------------------------------

    assert len(all_labels) == len(all_predictions)
    assert len(all_predictions) == len(test_loader.dataset)

    print("\nInference Complete")
    print("-"*40)
    print(f"Samples Evaluated : {len(all_labels)}")
    print(f"Predictions       : {len(all_predictions)}")
    print(f"Probability Rows  : {len(all_probabilities)}")
    print("-"*40)

    # -------------------------------------------------
    # Evaluation Metrics
    # -------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        all_labels,
        all_predictions
    )

    mcc = matthews_corrcoef(
        all_labels,
        all_predictions
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=le.classes_,
        digits=4,
        zero_division=0
    )
    cm = confusion_matrix(
        all_labels,
        all_predictions
    )
    print("\nEvaluation Metrics")
    print("=" * 60)

    print(f"Accuracy             : {accuracy:.4f}")
    print(f"Precision            : {precision:.4f}")
    print(f"Recall               : {recall:.4f}")
    print(f"F1 Score             : {f1:.4f}")
    print(f"Balanced Accuracy    : {balanced_acc:.4f}")
    print(f"Matthews CorrCoef    : {mcc:.4f}")

    print("=" * 60)
    print("\nClassification Report")
    print("-" * 60)

    print(report)

    print("\nCheckpoint Loaded Successfully")
    print("-" * 40)
    print(f"Model : {args.model}")
    print(f"Input Features : {input_dim}")
    print(f"Dataset : {args.dataset}")
    print(f"Classes : {num_classes}")
    print("-" * 40)

if __name__ == "__main__":
    main()



