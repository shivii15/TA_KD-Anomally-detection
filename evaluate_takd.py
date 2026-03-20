import torch
import torch.nn as nn
import joblib
import time
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from data.data_loader import preprocess_iot_data
from models.model import TeacherResNet, StudentMLP

def save_to_latex(y_true, y_pred, target_names, filename="results/metrics_table.tex"):
    """Generates an academic-grade LaTeX table for publication."""
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    
    # Selecting main metrics to avoid table overflow in double-column papers
    latex_string = df.to_latex(
        index=True, 
        column_format='|l|c|c|c|r|', 
        caption="Comparative Analysis of TGKD Student Performance on CIC-IoT-2023",
        label="table:tgkd_results",
        float_format="%.4f"
    )
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(latex_string)
    print(f"📄 LaTeX table saved to {filename}")

def evaluate_tgkd(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Evaluating on: {device}")

    # 1. Load Data (Memory-Safe)
    X, y, le, _ = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    input_dim = X.shape[1]
    num_classes = len(le.classes_)

    # 2. Create DataLoader to prevent OOM
    # A batch size of 4096 is safe for 48GB GPU with ~47 features
    test_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=4096, shuffle=False)

    # 3. Load Models
    teacher = TeacherResNet(input_dim, num_classes).to(device)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device, weights_only=False)['model_state_dict'])
    teacher.eval()

    student = StudentMLP(input_dim, num_classes).to(device)
    checkpoint = torch.load(args.student_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        student.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Student weights loaded from 'model_state_dict'")
    else:
        student.load_state_dict(checkpoint)
        print("✅ Student weights loaded directly")
    student.eval()

    # 4. Latency Benchmarking (Controlled environment)
    print("\n⏱️ Measuring Latency...")
    def get_latency(model, data_sample):
        model.eval()
        sample = data_sample[:100].to(device)
        start = time.time()
        with torch.no_grad():
            for _ in range(100): # Average over 100 runs for stability
                _ = model(sample)
        return (time.time() - start) / (100 * 100) * 1000 # ms per sample

    t_lat = get_latency(teacher, X_tensor)
    s_lat = get_latency(student, X_tensor)

    # 5. Batch-wise Prediction (The OOM Fix)
    print(f"📊 Generating Predictions for {len(X_tensor)} samples...")
    all_t_preds = []
    all_s_preds = []

    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            
            # Teacher Inference
            t_logits, _ = teacher(batch_X)
            all_t_preds.append(torch.argmax(t_logits, dim=1).cpu())

            # Student Inference (Unpack tuple)
            s_out = student(batch_X)
            s_logits = s_out[0] if isinstance(s_out, tuple) else s_out
            all_s_preds.append(torch.argmax(s_logits, dim=1).cpu())

    t_preds = torch.cat(all_t_preds).numpy()
    s_preds = torch.cat(all_s_preds).numpy()

    # 6. Metrics Calculation
    accuracy_t = (t_preds == y).mean()
    accuracy_s = (s_preds == y).mean()
    report = classification_report(y, s_preds, target_names=le.classes_)

    # 7. Comprehensive Reporting
    output_content = f"""
==================================================
        🏆 TGKD RESEARCH EVALUATION REPORT
==================================================
Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {args.data_path}
Samples Evaluated: {len(y)}
--------------------------------------------------
MODEL PERFORMANCE:
ResNet Teacher Accuracy: {accuracy_t:.4f}
TGKD Student Accuracy:   {accuracy_s:.4f}
Accuracy Retention:      {(accuracy_s/accuracy_t)*100:.2f}%

COMPUTATIONAL EFFICIENCY:
Teacher Latency: {t_lat:.4f} ms/sample
Student Latency: {s_lat:.4f} ms/sample
🚀 Speedup:      {t_lat/s_lat:.2f}x Faster
--------------------------------------------------
DETAILED STUDENT CLASSIFICATION REPORT:
{report}
==================================================
"""
    print(output_content)

    # Save Results
    os.makedirs("results", exist_ok=True)
    with open("results/tgkd_evaluation_results.txt", "w") as f:
        f.write(output_content)

    # Save LaTeX Table for Overleaf
    save_to_latex(y, s_preds, le.classes_, "results/student_performance_table_resnet.tex")
    
    print("✅ Evaluation Complete. Results and LaTeX tables are in the /results folder.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate TGKD Teacher-Student Framework")
    parser.add_argument('--data_path', type=str, required=True, help="Path to CIC-IoT CSVs")
    parser.add_argument('--teacher_path', type=str, required=True, help="Path to Teacher ResNet .pth")
    parser.add_argument('--student_path', type=str, required=True, help="Path to Student .pth")
    parser.add_argument('--iso_path', type=str, required=True, help="Path to Isolation Forest .pkl")
    parser.add_argument('--num_parts', type=int, default=5, help="Number of data parts to evaluate")
    
    args = parser.parse_args()
    evaluate_tgkd(args)