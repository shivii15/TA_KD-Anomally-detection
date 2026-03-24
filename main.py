import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import os
import joblib
import copy
from datetime import datetime

from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherDNN, TeacherResNet, StudentMLP


# ---------------------------
# ✅ SAFE GLOBALS FIX
# ---------------------------
import sklearn
torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.core.multiarray.scalar,
])

# ---------------------------
# ✅ SOTA UTILITY: EMA UPDATE
# ---------------------------
def update_ema_variables(model, ema_model, alpha=0.999):
    """Slowly updates the weights of the Mean Teacher student."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

# ---------------------------
# ✅ LOAD TEACHER MODEL
# ---------------------------
def load_teacher_model(args, input_dim, num_classes, device):
    if args.teacher_type == 'resnet':
        model = TeacherResNet(input_dim, num_classes)
        print("🏗️ Initialized Teacher: ResNet (v2)")
    else:
        model = TeacherDNN(input_dim, num_classes)
        print("🏗️ Initialized Teacher: Standard DNN (v1)")

    print(f"📂 Loading weights from: {args.teacher_path}")
    checkpoint = torch.load(args.teacher_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded from 'model_state_dict'")
    else:
        model.load_state_dict(checkpoint)
        print("ℹ️ Loaded raw state_dict")

    model.to(device)
    model.eval()
    return model

# ---------------------------
# ✅ TRUST SCORE FUNCTION
# ---------------------------
def compute_trust_score(teacher_probs, anomaly_scores, alpha=0.4, beta=0.3, gamma=0.3):
    conf, _ = torch.max(teacher_probs, dim=1)
    entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-9), dim=1)
    max_entropy = np.log(teacher_probs.shape[1])
    norm_entropy = entropy / max_entropy
    trust_score = (alpha * conf) + (beta * anomaly_scores) + (gamma * (1 - norm_entropy))
    return torch.clamp(trust_score, 0.1, 0.9) # SOTA tweak: avoid 0/1 extremes

# ---------------------------
# ✅ MAIN TRAINING PIPELINE
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="TGKD: Student Distillation Phase")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--teacher_type', type=str, default='resnet', choices=['dnn', 'resnet'])
    parser.add_argument('--iso_path', type=str, default="models/iso_forest_iot.pkl")
    parser.add_argument('--num_parts', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--temp_base', type=float, default=3.0)
    parser.add_argument('--feat_weight', type=float, default=0.5) # Higher weight for projected alignment
    parser.add_argument('--save_path', type=str, default="models/student_distilled.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # 1. Data Loading
    X, y, le, scaler = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    train_loader, val_loader = get_dataloaders(X, y, batch_size=args.batch_size)

    # 2. Load Models
    teacher = load_teacher_model(args, X.shape[1], len(le.classes_), device)
    student = StudentMLP(X.shape[1], len(le.classes_)).to(device)
    
    # --- EMA Initialization ---
    ema_student = copy.deepcopy(student)
    for param in ema_student.parameters():
        param.requires_grad = False

    # 3. Load Anomaly Model
    if not os.path.exists(args.iso_path):
        raise FileNotFoundError(f"❌ Anomaly Model missing at {args.iso_path}")
    iso_model = joblib.load(args.iso_path)

    # Use AdamW for better weight decay handling (Research Standard)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # -------- Teacher Forward --------
            with torch.no_grad():
                t_logits, t_feat = teacher(batch_x)
                t_probs = torch.softmax(t_logits, dim=1)
                raw_a = torch.tensor(iso_model.decision_function(batch_x.cpu().numpy()), device=device)
                norm_a = (raw_a - raw_a.min()) / (raw_a.max() - raw_a.min() + 1e-9)
                trust_score = compute_trust_score(t_probs, norm_a)
                tau_adapt = args.temp_base * (2.0 - trust_score)

            # -------- Student Forward --------
            s_logits, s_feat = student(batch_x)

            # 1. Gated Cross-Entropy
            loss_ce = F.cross_entropy(s_logits, batch_y, reduction='none')
            loss_ce = ((1 - trust_score) * loss_ce).mean()

            # 2. Gated KD Loss (Logits)
            tau_expand = tau_adapt.unsqueeze(1)
            soft_t = torch.softmax(t_logits / tau_expand, dim=1)
            soft_s = torch.log_softmax(s_logits / tau_expand, dim=1)
            loss_kd = F.kl_div(soft_s, soft_t, reduction='none').sum(dim=1)
            loss_kd = (trust_score * loss_kd * (tau_adapt ** 2)).mean()

            # 3. Projected Feature Alignment (L_feat)
            loss_feat = F.mse_loss(s_feat, t_feat)

            # 4. SOTA Addition: Consistency Loss (EMA)
            with torch.no_grad():
                ema_logits, _ = ema_student(batch_x)
            loss_consist = F.mse_loss(s_logits, ema_logits)

            # Total Loss
            total_loss = loss_ce + loss_kd + (args.feat_weight * loss_feat) + (0.1 * loss_consist)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update EMA "Mean Teacher"
            update_ema_variables(student, ema_student)
            epoch_loss += total_loss.item()

        print(f"Epoch [{epoch}/{args.epochs}] | TGKD-SOTA Loss: {epoch_loss / len(train_loader):.4f}")

    # 5. Save Model
    torch.save({
        'model_state_dict': student.state_dict(),
        'classes': le.classes_,
        'metadata': {
            'method': 'TGKD-SOTA',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }, args.save_path)

    print(f"\n✅ TGKD-SOTA Student saved to {args.save_path}")

if __name__ == "__main__":
    main()