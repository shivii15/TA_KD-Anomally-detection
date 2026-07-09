import argparse
import os
import json
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from models.trust_gate import TGKDTrustModule
from data.data_loader import load_dataset, get_dataloaders
from config.default_config import DEFAULT_CONFIG
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
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
        "--iso_path",
        type=str,
        required=True,
        help="Isolation Forest checkpoint"
    )

    parser.add_argument(
        "--temp_base",
        type=float,
        default=3.0
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
    parser.add_argument('--resnet_path', type=str, required=True, help="Path to trained ResNet teacher")
    parser.add_argument('--trans_path', type=str, required=True, help="Path to trained Transformer teacher")
    parser.add_argument('--lstm_path', type=str, required=True, help="Path to trained LSTM teacher")
    
    return parser.parse_args()

def load_teacher(model_class, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model = model_class(
        checkpoint["input_dim"],
        checkpoint["num_classes"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint

def main():

    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = (
        f"{args.model}_{args.dataset}_{timestamp}"
    )

    result_dir = os.path.join(
        "results",
        experiment_name
    )

    os.makedirs(result_dir, exist_ok=True)
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
    print("\nCheckpoint Loaded Successfully")

    iso_package = joblib.load(args.iso_path)
    iso_model = (
        iso_package["model"]
        if isinstance(iso_package, dict)
        else iso_package
    )

    print("✅ Isolation Forest loaded.")
    trust_module = TGKDTrustModule(
        isolation_forest=iso_model,
        base_temperature=args.temp_base
    )

    # Initialize Committee of Experts

    t_resnet, resnet_ckpt = load_teacher(
        TeacherResNet,
        args.resnet_path
    )

    t_trans, transformer_ckpt = load_teacher(
        TeacherTransformer,
        args.trans_path
    )

    t_lstm, lstm_ckpt = load_teacher(
        TeacherLSTM,
        args.lstm_path
    )

    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_logits = []
    all_features = []
    all_trust = []
    all_confidence = []
    all_entropy = []
    all_temperature = []
    all_disagreement = []
    all_anomaly = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits, features = model(batch_x)

            logits_r, feat_r = t_resnet(batch_x)
            logits_t, _ = t_trans(batch_x)
            logits_l, _ = t_lstm(batch_x)
            # Committee aggregation
            committee_logits = (
                logits_r +
                logits_t +
                logits_l
            ) / 3.0

            trust_outputs = trust_module(
                teacher_logits=committee_logits,
                student_logits=logits,
                x_input=batch_x
            )

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
            all_trust.append(trust_outputs["trust_score"].cpu())
            all_confidence.append(trust_outputs["confidence"].cpu())
            all_entropy.append(trust_outputs["entropy_trust"].cpu())
            all_temperature.append(trust_outputs["temperature"].cpu())
            all_disagreement.append(trust_outputs["disagreement"].cpu())
            all_anomaly.append(trust_outputs["anomaly_score"].cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_predictions = torch.cat(all_predictions).numpy()
    all_probabilities = torch.cat(all_probabilities).numpy()
    all_logits = torch.cat(all_logits).numpy()
    all_features = torch.cat(all_features).numpy()
    all_trust = torch.cat(all_trust).numpy()
    all_confidence = torch.cat(all_confidence).numpy()
    all_entropy = torch.cat(all_entropy).numpy()
    all_temperature = torch.cat(all_temperature).numpy()
    all_disagreement = torch.cat(all_disagreement).numpy()
    all_anomaly = torch.cat(all_anomaly).numpy()

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

    metrics = {

        "dataset": args.dataset,
        "model": args.model,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "balanced_accuracy": float(balanced_acc),
        "mcc": float(mcc),
        "num_samples": int(len(all_labels))
    }
    with open(
        os.path.join(result_dir, "metrics.json"),
        "w"
    ) as f:
        json.dump(metrics, f, indent=4)
    with open(
        os.path.join(result_dir, "classification_report.txt"),
        "w"
    ) as f:
        f.write(report)
    prediction_df = pd.DataFrame({
        "True Label": all_labels,
        "Predicted Label": all_predictions
    })
    prediction_df.to_csv(
        os.path.join(result_dir, "predictions.csv"),
        index=False
    )
    np.save(
        os.path.join(result_dir, "trust.npy"),
        all_trust
    )
    np.save(
        os.path.join(result_dir, "confidence.npy"),
        all_confidence
    )
    np.save(
        os.path.join(result_dir, "entropy.npy"),
        all_entropy
    )
    np.save(
        os.path.join(result_dir, "temperature.npy"),
        all_temperature
    )
    np.save(
        os.path.join(result_dir, "disagreement.npy"),
        all_disagreement
    )
    np.save(
        os.path.join(result_dir, "anomaly.npy"),
        all_anomaly
    )
    np.save(
        os.path.join(result_dir, "probabilities.npy"),
        all_probabilities
    )
    np.save(
        os.path.join(result_dir, "logits.npy"),
        all_logits
    )
    np.save(
        os.path.join(result_dir, "features.npy"),
        all_features
    )
    experiment = {
        "timestamp": timestamp,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "model": args.model,
        "device": str(device),
        "samples": int(len(all_labels)),
        "classes": list(le.classes_)
    }
    with open(
        os.path.join(result_dir, "inference_info.json"),
        "w"
    ) as f:
        json.dump(experiment, f, indent=4)

    np.save(
        os.path.join(result_dir, "confusion_matrix.npy"),
        cm
    )
    np.save(
        os.path.join(result_dir, "labels.npy"),
        all_labels
    )
    np.save(
        os.path.join(result_dir, "predictions.npy"),
        all_predictions
    )
    plt.figure(figsize=(10,8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            result_dir,
            "confusion_matrix.png"
        ),
        dpi=300
    )
    plt.close()

    print("\nResults saved successfully")
    print("=" * 60)
    print(result_dir)
    print("=" * 60)

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



