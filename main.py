import torch
import torch.nn as nn
import torch.optim as optim
from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import TeacherDNN, TeacherResNet, StudentMLP
import argparse
import os
from datetime import datetime

def load_teacher_model(args, input_dim, num_classes, device):
    """Factory to initialize the correct Teacher and load its weights."""
    if args.teacher_type == 'resnet':
        model = TeacherResNet(input_dim, num_classes)
        print("🏗️ Initialized Teacher: ResNet (v2)")
    else:
        model = TeacherDNN(input_dim, num_classes)
        print("🏗️ Initialized Teacher: Standard DNN (v1)")

    print(f"📂 Loading weights from: {args.teacher_path}")
    checkpoint = torch.load(args.teacher_path, map_location=device)

    # Handle both metadata-wrapped checkpoints and raw state_dicts
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        meta = checkpoint.get('metadata', {})
        print(f"🏷️ Teacher Version: {meta.get('version', 'N/A')} | Trained on: {meta.get('timestamp', 'Unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval() # Teacher is always in eval mode
    return model

def main():
    parser = argparse.ArgumentParser(description="TGKD: Student Distillation Phase")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--teacher_type', type=str, default='resnet', choices=['dnn', 'resnet'])
    parser.add_argument('--num_parts', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--temperature', type=float, default=3.0, help="Distillation temperature")
    parser.add_argument('--alpha', type=float, default=0.5, help="Weight for KD loss vs Student loss")
    parser.add_argument('--save_path', type=str, default="models/student_distilled.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data Loading (4-item Unpack)
    print(f"📊 Loading CIC-IoT-2023 data...")
    X, y, le, scaler = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    train_loader, val_loader = get_dataloaders(X, y, batch_size=args.batch_size)

    input_dim = X.shape[1]
    num_classes = len(le.classes_)

    # 2. Initialize Models
    teacher = load_teacher_model(args, input_dim, num_classes, device)
    student = StudentMLP(input_dim, num_classes).to(device)

    # 3. Distillation Components
    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')

    print(f"🚀 Starting Distillation on {device}...")

    # 4. Distillation Loop
    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Teacher Soft Targets
            with torch.no_grad():
                teacher_logits = teacher(batch_x)
            
            # Student Predictions
            student_logits = student(batch_x)
            
            # --- TGKD Loss Calculation ---
            # Standard Loss (Hard Labels)
            loss_cls = criterion_cls(student_logits, batch_y)
            
            # Distillation Loss (Soft Labels with Temperature)
            soft_teacher = torch.softmax(teacher_logits / args.temperature, dim=1)
            soft_student = torch.log_softmax(student_logits / args.temperature, dim=1)
            loss_kd = criterion_kd(soft_student, soft_teacher) * (args.temperature ** 2)
            
            # Combined Loss
            loss = (args.alpha * loss_cls) + ((1 - args.alpha) * loss_kd)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch}/{args.epochs}] | Avg Distill Loss: {total_loss/len(train_loader):.4f}")

    # 5. Save Distilled Student
    torch.save({
        'model_state_dict': student.state_dict(),
        'le': le,
        'scaler': scaler,
        'metadata': {
            'parent_teacher': args.teacher_type,
            'temperature': args.temperature,
            'alpha': args.alpha,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }, args.save_path)
    print(f"✅ Student model successfully distilled and saved to {args.save_path}")

if __name__ == "__main__":
    main()