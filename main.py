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

from data.data_loader import load_dataset, get_dataloaders
from models.model import TeacherResNet, TeacherTransformer, TeacherLSTM, StudentMLP

# ---------------------------
# ✅ SOTA UTILITIES
# ---------------------------
def update_ema_variables(model, ema_model, alpha=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def normalize_anomaly_score(raw_scores):
    """Sigmoid normalization for Isolation Forest [-0.5, 0.5] -> [0, 1]"""
    return torch.sigmoid(raw_scores * 10.0)

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
    parser.add_argument('--num_parts', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--temp_base', type=float, default=3.0)
    parser.add_argument('--feat_weight', type=float, default=0.5)
    parser.add_argument('--save_name', type=str, default="Student_MultiTeacher_TGKD")
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
    best_student_path = f"models/{args.save_name}_{timestamp}_best.pth"
    log_path = f"logs/{args.save_name}_{timestamp}_history.json"

    # 2. Load Data
    #X, y, le, _ = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    X, y, le, scaler = load_dataset(args)
    train_loader, val_loader = get_dataloaders(X, y, batch_size=args.batch_size)
    num_classes = len(le.classes_)
    input_dim = X.shape[1]

    # 3. Initialize Committee of Experts
    def load_teacher(model_class, path):
        m = model_class(input_dim, num_classes)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        return m.to(device).eval()

    t_resnet = load_teacher(TeacherResNet, args.resnet_path)
    t_trans = load_teacher(TeacherTransformer, args.trans_path)
    t_lstm = load_teacher(TeacherLSTM, args.lstm_path)
    print("🎓 Committee of Experts Loaded: ResNet, Transformer, LSTM")

    # Load Anomaly Detector
    iso_package = joblib.load(args.iso_path)
    iso_model = iso_package['model'] if isinstance(iso_package, dict) else iso_package
    print(f"🌲 Anomaly Module Active")

    # Initialize Student
    student = StudentMLP(input_dim, num_classes).to(device)
    ema_student = copy.deepcopy(student)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    # 4. Training Loop
    history = {"train_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"🚀 Multi-Distill E{epoch}", leave=False)

        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            with torch.no_grad():
                # Get expert opinions
                logits_r, feat_r = t_resnet(batch_x)
                logits_t, _ = t_trans(batch_x)
                logits_l, _ = t_lstm(batch_x)
                
                # Multi-Teacher Aggregation (Average Logits)
                avg_t_logits = (logits_r + logits_t + logits_l) / 3.0
                
                # Trust-Gating Logic
                raw_a = torch.tensor(iso_model.decision_function(batch_x.cpu().numpy()), device=device, dtype=torch.float32)
                norm_a = normalize_anomaly_score(raw_a)
                
                t_probs = torch.softmax(avg_t_logits, dim=1)
                trust_score = (0.4 * torch.max(t_probs, 1)[0]) + (0.6 * norm_a)
                tau_adapt = (args.temp_base * (2.0 - trust_score)).unsqueeze(1)

            # Student Forward
            s_logits, s_feat = student(batch_x)

            # ---------------------------
            # ✅ MULTI-TEACHER LOSS
            # ---------------------------
            # 1. Trust-Gated Cross Entropy (Hard Loss)
            loss_ce = ( (1 - trust_score) * F.cross_entropy(s_logits, batch_y, reduction='none') ).mean()
            
            # 2. Adaptive Soft Distillation (Soft Loss)
            soft_t = torch.softmax(avg_t_logits / tau_adapt, dim=1)
            soft_s = torch.log_softmax(s_logits / tau_adapt, dim=1)
            loss_kd = (trust_score * F.kl_div(soft_s, soft_t, reduction='none').sum(1) * (tau_adapt.squeeze()**2)).mean()
            
            # 3. Feature Alignment (Against SOTA ResNet Expert)
            loss_feat = args.feat_weight * F.mse_loss(s_feat, feat_r)
            
            batch_loss = loss_ce + loss_kd + loss_feat
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            update_ema_variables(student, ema_student)
            total_loss += batch_loss.item()
            train_pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

        # Validation & Logging
        val_acc = validate(student, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        with open(log_path, 'w') as f:
            json.dump(history, f, indent=4)

        print(f"✅ E{epoch}: Loss {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': student.state_dict(),
                'ema_state_dict': ema_student.state_dict(),
                'classes': le.classes_,
                'val_acc': best_acc
            }, best_student_path)
            print(f"⭐ New Best Multi-Teacher Student: {val_acc:.2f}%")

if __name__ == "__main__":
    main()