import torch
import torch.nn as nn
import joblib
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from data.data_loader import preprocess_iot_data
from models.model import TeacherResNet, StudentMLP # Ensure these match your filenames

def evaluate_tgkd(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Evaluating on: {device}")

    # 1. Load Data (Using the final parts for testing)
    X, y, le, _ = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    X_test = torch.tensor(X, dtype=torch.float32).to(device)
    y_test = torch.tensor(y, dtype=torch.long).to(device)
    
    input_dim = X.shape[1]
    num_classes = len(le.classes_)

    # 2. Load Models
    # Load ResNet Teacher
    teacher = TeacherResNet(input_dim, num_classes).to(device)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device, weights_only=False)['model_state_dict'])
    teacher.eval()

    # Load Distilled Student
    student = StudentMLP(input_dim, num_classes).to(device)
    student.load_state_dict(torch.load(args.student_path, map_location=device, weights_only=False))
    student.eval()

    # Load Isolation Forest (Gating Module)
    iso_forest = joblib.load(args.iso_path)

    # 3. Performance & Latency Testing
    print("\n⏱️ Measuring Latency...")
    
    def get_latency(model, data):
        start = time.time()
        with torch.no_grad():
            _ = model(data[:100]) # Benchmark on 100 samples
        return (time.time() - start) / 100 * 1000

    t_lat = get_latency(teacher, X_test)
    s_lat = get_latency(student, X_test)

    # 4. Final Predictions
    print("📊 Generating Predictions...")
    with torch.no_grad():
        t_logits, _ = teacher(X_test)
        t_preds = torch.argmax(t_logits, dim=1).cpu().numpy()
        
        s_logits = student(X_test)
        s_preds = torch.argmax(s_logits, dim=1).cpu().numpy()

    # 5. Output Report
    print("\n" + "="*30)
    print("🏆 FINAL RESEARCH REPORT")
    print("="*30)
    print(f"ResNet Teacher Accuracy: {(t_preds == y).mean():.4f}")
    print(f"TGKD Student Accuracy:   {(s_preds == y).mean():.4f}")
    print("-" * 30)
    print(f"Teacher Latency: {t_lat:.4f} ms/sample")
    print(f"Student Latency: {s_lat:.4f} ms/sample")
    print(f"🚀 Speedup: {t_lat/s_lat:.2f}x Faster")
    print("="*30)

    print("\nDetailed Student Classification Report:")
    print(classification_report(y, s_preds, target_names=le.classes_))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--student_path', type=str, required=True)
    parser.add_argument('--iso_path', type=str, required=True)
    parser.add_argument('--num_parts', type=int, default=5)
    args = parser.parse_args()
    evaluate_tgkd(args)