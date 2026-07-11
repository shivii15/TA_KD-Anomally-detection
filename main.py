import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import os
import joblib
import copy
import json
from datetime import datetime
from tqdm import tqdm
from config.default_config import DEFAULT_CONFIG
from datetime import datetime
from data.data_loader import load_dataset, get_dataloaders
from models.model import (
    TeacherResNet,
    TeacherTransformer,
    TeacherLSTM,
    StudentMLP
)

from models.trust_gate import TGKDTrustModule
# ---------------------------
# ✅ SOTA UTILITIES
# ---------------------------
def update_ema_variables(model, ema_model, alpha=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits, _ = model(batch_x)
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    return 100. * correct / total

# ---------------------------
# ✅ MAIN PIPELINE
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-Teacher TGKD: Student Distillation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cic",
        choices=["cic", "nbaiot"],
        help="Dataset to use"
    )
    #parser.add_argument('--data_path', type=str, required=True)
    
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Dataset directory (optional for N-BaIoT)"
    )
    # Teacher Paths
    parser.add_argument('--resnet_path', type=str, required=True, help="Path to trained ResNet teacher")
    parser.add_argument('--trans_path', type=str, required=True, help="Path to trained Transformer teacher")
    parser.add_argument('--lstm_path', type=str, required=True, help="Path to trained LSTM teacher")
    
    parser.add_argument('--iso_path', type=str, default="models/iso_forest_iot.pkl")
    parser.add_argument('--num_parts', type=int, default=DEFAULT_CONFIG["num_parts"])
    #parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["epochs"]
    )
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument('--temp_base', type=float, default=DEFAULT_CONFIG["temp_base"])
    parser.add_argument('--feat_weight', type=float, default=DEFAULT_CONFIG["feat_weight"])
    parser.add_argument('--save_name', type=str, default="Student_MultiTeacher_TGKD")

    parser.add_argument(
        "--disable_confidence",
        action="store_true",
        help="Disable confidence component."
    )

    parser.add_argument(
        "--disable_entropy",
        action="store_true",
        help="Disable entropy component."
    )

    parser.add_argument(
        "--disable_anomaly",
        action="store_true",
        help="Disable anomaly component."
    )

    parser.add_argument(
        "--disable_disagreement",
        action="store_true",
        help="Disable disagreement component."
    )

    parser.add_argument(
        "--disable_temperature",
        action="store_true",
        help="Disable adaptive temperature."
    )

    parser.add_argument(
        "--disable_feature",
        action="store_true",
        help="Disable feature alignment."
    )

    args = parser.parse_args()

    if args.dataset == "cic" and args.data_path is None:
        raise ValueError(
            "--data_path is required when using the CIC-IoT dataset."
        )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # 1. Setup Logging & Paths
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)



    # --------------------------------------------------
    # Experiment Directories
    # --------------------------------------------------

    experiment_name = (
        f"{args.save_name}_"
        f"{args.dataset}_"
        f"{timestamp}"
    )

    experiment_dir = os.path.join(
        "experiments",
        experiment_name
    )

    checkpoint_dir = os.path.join(
        experiment_dir,
        "checkpoints"
    )

    log_dir = os.path.join(
        experiment_dir,
        "logs"
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --------------------------------------------------
    # Checkpoint Paths
    # --------------------------------------------------

    best_student_path = os.path.join(
        checkpoint_dir,
        "best.pth"
    )

    last_student_path = os.path.join(
        checkpoint_dir,
        "last.pth"
    )

    # --------------------------------------------------
    # Training Log
    # --------------------------------------------------

    log_path = os.path.join(
        log_dir,
        "history.json"
    )



    # 2. Load Data
    #X, y, le, _ = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    X, y, le, scaler = load_dataset(args)
    train_loader, val_loader,_ = get_dataloaders(X, y, batch_size=args.batch_size)
    num_classes = len(le.classes_)
    input_dim = X.shape[1]

    # 3. Initialize Committee of Experts
    def load_teacher(model_class, checkpoint_path):

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

    trust_module = TGKDTrustModule(
        confidence_weight=0.4,
        anomaly_weight=0.3,
        entropy_weight=0.2,
        disagreement_weight=0.1,
        base_temperature=args.temp_base,
        enable_confidence=not args.disable_confidence,
        enable_entropy=not args.disable_entropy,
        enable_anomaly=not args.disable_anomaly,
        enable_disagreement=not args.disable_disagreement,
        enable_temperature=not args.disable_temperature
    )

    trust_module.load_anomaly_detector(
        args.iso_path
    )

    teacher_checkpoints = [
        resnet_ckpt,
        transformer_ckpt,
        lstm_ckpt
    ]

    for checkpoint in teacher_checkpoints:

        if checkpoint["dataset"] != args.dataset:

            raise ValueError(
                f"Teacher trained on {checkpoint['dataset']} "
                f"cannot be used with {args.dataset}."
            )

    print("🎓 Committee of Experts Loaded: ResNet, Transformer, LSTM")

    # Load Anomaly Detector
    print(f"🌲 Anomaly Module Active")

    print("\nExperiment Configuration")
    print("=" * 50)
    print(f"Dataset            : {args.dataset}")
    print(f"Student            : StudentMLP")
    print(f"Teacher Committee  : ResNet + Transformer + LSTM")
    print(f"Isolation Forest   : {args.iso_path}")
    print(f"Epochs             : {args.epochs}")
    print(f"Batch Size         : {args.batch_size}")
    print(f"Learning Rate      : {args.lr}")
    print(f"Feature Weight     : {args.feat_weight}")
    print(f"Base Temperature   : {args.temp_base}")
    print("=" * 50)

    # Initialize Student
    student = StudentMLP(input_dim, num_classes).to(device)
    ema_student = copy.deepcopy(student)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    # 4. Training Loop
    history = {
        "epoch": [],
        "train_loss": [],
        "ce_loss": [],
        "kd_loss": [],
        "feature_loss": [],
        "val_acc": [],
        "confidence": [],
        "entropy": [],
        "anomaly": [],
        "disagreement": [],
        "trust_score": [],
        "temperature": [],
        "lr": [],
    }
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0
        epoch_ce_loss = 0
        epoch_kd_loss = 0
        epoch_feature_loss = 0
        epoch_confidence = 0
        epoch_entropy = 0
        epoch_anomaly = 0
        epoch_disagreement = 0
        epoch_trust = 0
        epoch_temperature = 0

        train_pbar = tqdm(train_loader, desc=f"🚀 Multi-Distill E{epoch}", leave=False)

        for i, (batch_x, batch_y) in enumerate(train_pbar):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            with torch.no_grad():
                # Get expert opinions
                logits_r, feat_r = t_resnet(batch_x)
                logits_t, _ = t_trans(batch_x)
                logits_l, _ = t_lstm(batch_x)
                # Committee aggregation
                committee_logits = (
                    logits_r +
                    logits_t +
                    logits_l
                ) / 3.0

            # -------------------------
            # Student Forward
            # -------------------------

            s_logits, s_feat = student(batch_x)
            trust_outputs = trust_module(
                teacher_logits=committee_logits,
                student_logits=s_logits.detach(),
                x_input=batch_x
            )

            # -------------------------
            # Trust Module
            # -------------------------
            trust_score = trust_outputs["trust_score"]
            temperature = trust_outputs["temperature"]
            confidence = trust_outputs["confidence"]
            entropy = trust_outputs["entropy_trust"]
            anomaly = trust_outputs["anomaly_score"]
            disagreement = trust_outputs["disagreement"]

            # Print only the first batch
            if epoch == 1 and i == 0:

                print("\n========== Trust Module ==========")
                print(f"Confidence    : {confidence.mean().item():.4f}")
                print(f"Entropy Trust : {entropy.mean().item():.4f}")
                print(f"Anomaly Score : {anomaly.mean().item():.4f}")
                print(f"Disagreement  : {disagreement.mean().item():.4f}")
                print(f"Trust Score   : {trust_score.mean().item():.4f}")
                print(f"Temperature   : {temperature.mean().item():.4f}")
                print("==================================\n")

            # --------------------------------------------------
            # Loss Calculation
            # --------------------------------------------------

            # Cross Entropy
            loss_ce = (
                (1 - trust_score)
                * F.cross_entropy(
                    s_logits,
                    batch_y,
                    reduction="none"
                )
            ).mean()

            # Knowledge Distillation
            temperature_kd = temperature.unsqueeze(1)

            soft_teacher = torch.softmax(
                committee_logits / temperature_kd,
                dim=1
            )

            soft_student = torch.log_softmax(
                s_logits / temperature_kd,
                dim=1
            )

            loss_kd = (
                trust_score
                * F.kl_div(
                    soft_student,
                    soft_teacher,
                    reduction="none"
                ).sum(1)
                * (temperature_kd.squeeze() ** 2)
            ).mean()

            #--------------------
            #  Feature Alignment
            #--------------------
            #loss_feat = (
            #    args.feat_weight
            #    * F.mse_loss(s_feat, feat_r)
            #)

            if args.disable_feature:

                loss_feat = torch.tensor(
                    0.0,
                    device=device
                )

            else:

                loss_feat = (
                    args.feat_weight *
                    F.mse_loss(
                        s_feat,
                        feat_r
                    )
                )

            # Total Loss
            batch_loss = (
                loss_ce
                + loss_kd
                + loss_feat
            )

            # --------------------------------------------------
            # Update Epoch Statistics
            # --------------------------------------------------

            epoch_ce_loss += loss_ce.item()
            epoch_kd_loss += loss_kd.item()
            epoch_feature_loss += loss_feat.item()

            epoch_confidence += confidence.mean().item()
            epoch_entropy += entropy.mean().item()
            epoch_anomaly += anomaly.mean().item()
            epoch_disagreement += disagreement.mean().item()
            epoch_trust += trust_score.mean().item()
            epoch_temperature += temperature.mean().item()

            # --------------------------------------------------
            # Optimization
            # --------------------------------------------------

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            update_ema_variables(student, ema_student)
            total_loss += batch_loss.item()
            train_pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

        # Validation & Logging
        val_acc = validate(student, val_loader, device)
        avg_loss = total_loss / len(train_loader)

        num_batches = len(train_loader)
        avg_ce_loss = epoch_ce_loss / num_batches
        avg_kd_loss = epoch_kd_loss / num_batches
        avg_feature_loss = epoch_feature_loss / num_batches
        avg_confidence = epoch_confidence / num_batches
        avg_entropy = epoch_entropy / num_batches
        avg_anomaly = epoch_anomaly / num_batches
        avg_disagreement = epoch_disagreement / num_batches
        avg_trust = epoch_trust / num_batches
        avg_temperature = epoch_temperature / num_batches

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["ce_loss"].append(avg_ce_loss)
        history["kd_loss"].append(avg_kd_loss)
        history["feature_loss"].append(avg_feature_loss)
        history["val_acc"].append(val_acc)
        history["confidence"].append(avg_confidence)
        history["entropy"].append(avg_entropy)
        history["anomaly"].append(avg_anomaly)
        history["disagreement"].append(avg_disagreement)
        history["trust_score"].append(avg_trust)
        history["temperature"].append(avg_temperature)
        history["lr"].append(
            optimizer.param_groups[0]["lr"]
        )

        with open(log_path, 'w') as f:
            json.dump(history, f, indent=4)

        print("\n" + "=" * 65)
        print(f"Epoch {epoch:03d}/{args.epochs}")
        print("-" * 65)
        print(f"Train Loss      : {avg_loss:.4f}")
        print(f"CE Loss         : {avg_ce_loss:.4f}")
        print(f"KD Loss         : {avg_kd_loss:.4f}")
        print(f"Feature Loss    : {avg_feature_loss:.4f}")
        print(f"Validation Acc  : {val_acc:.2f}%")
        print(f"Trust Score     : {avg_trust:.4f}")
        print(f"Temperature     : {avg_temperature:.4f}")
        print(f"Confidence      : {avg_confidence:.4f}")
        print(f"Entropy         : {avg_entropy:.4f}")
        print(f"Anomaly         : {avg_anomaly:.4f}")
        print(f"Disagreement    : {avg_disagreement:.4f}")
        print("=" * 65)

        checkpoint = {
            "dataset": args.dataset,
            "student": "StudentMLP",
            "teacher_committee": [
                "TeacherResNet",
                "TeacherTransformer",
                "TeacherLSTM"
            ],
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "ema_state_dict": ema_student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": val_acc,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "classes": le.classes_
        }
        torch.save(
            checkpoint,
            last_student_path
        )

        if val_acc > best_acc:

            print(">>> Saving new best student...")

            best_acc = val_acc

            checkpoint["best_val_acc"] = best_acc

            torch.save(
                checkpoint,
                best_student_path
            )

            print(f"⭐ New Best Student : {best_acc:.2f}%")
            print(f"💾 Saved to: {best_student_path}")

if __name__ == "__main__":
    main()