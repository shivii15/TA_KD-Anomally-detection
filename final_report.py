import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import joblib
import numpy as np
import argparse
from sklearn.metrics import classification_report, accuracy_score, f1_score
from models.model import TeacherResNet, TeacherDNN, StudentMLP
from data.data_loader import preprocess_iot_data, get_dataloaders

def fgsm_attack(data, epsilon, data_grad):
    """Perturb data for robustness testing."""
    sign_data_grad = data_grad.sign()
    perturbed_data = data + epsilon * sign_data_grad
    return torch.clamp(perturbed_data, 0, 1)

def get_model_size(model, path="temp.pth"):
    """Calculate physical disk size in MB."""
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return size

def evaluate_full_metrics(model, loader, device, name, epsilon=0.05):
    model.eval()
    all_preds, all_labels = [], []
    adv_correct, total = 0, 0
    
    # 1. Accuracy & Robustness (Batch Processing)
    start_time = time.time()
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Standard Inference
        with torch.no_grad():
            logits, _ = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        # Adversarial Attack (Batch-level FGSM)
        batch_x.requires_grad = True
        outputs, _ = model(batch_x)
        loss = F.cross_entropy(outputs, batch_y)
        model.zero_grad()
        loss.backward()
        
        perturbed_data = fgsm_attack(batch_x, epsilon, batch_x.grad.data)
        with torch.no_grad():
            adv_outputs, _ = model(perturbed_data)
            adv_preds = adv_outputs.max(1, keepdim=True)[1]
            adv_correct += adv_preds.eq(batch_y.view_as(adv_preds)).sum().item()
        total += batch_y.size(0)
    
    # 2. Efficiency Metrics (Single-Sample Latency)
    dummy_input = torch.randn(1, batch_x.shape[1]).to(device)
    latencies = []
    for _ in range(100):
        t0 = time.time()
        _ = model(dummy_input)
        latencies.append(time.time() - t0)
    avg_latency = np.mean(latencies) * 1000 # ms
    
    # Compile Results
    results = {
        "name": name,
        "acc": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average='weighted'),
        "adv_acc": adv_correct / total,
        "latency": avg_latency,
        "params": sum(p.numel() for p in model.parameters()),
        "size": get_model_size(model)
    }
    return results

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Initializing Final Evaluation on {device}...")

    # Load Held-out Test Data (Using parts 160-169)
    X, y, le, _ = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    _, test_loader = get_dataloaders(X, y, batch_size=args.batch_size)
    
    input_dim, num_classes = X.shape[1], len(le.classes_)

    # Define Model Configurations
    configs = [
        (TeacherDNN(input_dim, num_classes), args.tdnn_path, "Teacher DNN (V1)"),
        (TeacherResNet(input_dim, num_classes), args.tres_path, "Teacher ResNet (V2)"),
        (StudentMLP(input_dim, num_classes), args.stud_path, "TGKD Student (Full)")
    ]

    final_table = []
    for model, path, name in configs:
        if os.path.exists(path):
            print(f"⌛ Evaluating {name}...")
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
            model.to(device)
            final_table.append(evaluate_full_metrics(model, test_loader, device, name, args.epsilon))
        else:
            print(f"⚠️ Skipping {name}: Weights not found at {path}")

    # --- Print FINAL TABLES for Paper ---
    print("\n" + "="*85)
    print(f"{'Model Name':<25} | {'Acc (%)':<8} | {'Adv Acc':<8} | {'Latency':<10} | {'Size (MB)':<8}")
    print("-" * 85)
    for r in final_table:
        print(f"{r['name']:<25} | {r['acc']*100:<8.2f} | {r['adv_acc']:<8.4f} | {r['latency']:<7.4f} ms | {r['size']:<8.4f}")
    print("="*85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--tdnn_path', type=str, default="models/teacher_dnn_best.pth")
    parser.add_argument('--tres_path', type=str, default="models/teacher_resnet_best.pth")
    parser.add_argument('--stud_path', type=str, default="models/student_tgkd_final.pth")
    parser.add_argument('--num_parts', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epsilon', type=float, default=0.05, help="Attack strength")
    args = parser.parse_args()
    main(args)