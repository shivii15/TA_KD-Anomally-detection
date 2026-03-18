import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from models.model import TeacherDNN, TeacherResNet, StudentMLP
from data.data_loader import preprocess_iot_data, get_dataloaders

def fgsm_attack(data, epsilon, data_grad):
    """Generates perturbed data for robustness testing."""
    sign_data_grad = data_grad.sign()
    perturbed_data = data + epsilon * sign_data_grad
    return torch.clamp(perturbed_data, 0, 1)

def run_evaluation(model, loader, device, name, epsilon=0.05):
    model.eval()
    all_preds = []
    all_labels = []
    adv_correct = 0
    total = 0
    
    # 1. Standard Performance & Adversarial Test
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Standard Inference
        with torch.no_grad():
            logits, _ = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        # Adversarial Attack (FGSM)
        batch_x.requires_grad = True
        outputs, _ = model(batch_x)
        loss = F.cross_entropy(outputs, batch_y)
        model.zero_grad()
        loss.backward()
        
        data_grad = batch_x.grad.data
        perturbed_data = fgsm_attack(batch_x, epsilon, data_grad)
        
        with torch.no_grad():
            adv_outputs, _ = model(perturbed_data)
            adv_preds = adv_outputs.max(1, keepdim=True)[1]
            adv_correct += adv_preds.eq(batch_y.view_as(adv_preds)).sum().item()
        
        total += batch_y.size(0)

    # 2. Efficiency Metrics
    dummy_input = torch.randn(1, batch_x.shape[1]).to(device)
    start_time = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    latency = ((time.time() - start_time) / 100) * 1000 # ms
    
    params = sum(p.numel() for p in model.parameters())
    
    print(f"\n--- Results for {name} ---")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Adversarial Accuracy (eps={epsilon}): {adv_correct/total:.4f}")
    print(f"Latency: {latency:.4f} ms | Params: {params:,}")
    return accuracy_score(all_labels, all_preds)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Test Data (Using parts 160-169 as the held-out test set)
    X, y, le, scaler = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    _, test_loader = get_dataloaders(X, y, batch_size=args.batch_size)
    
    input_dim = X.shape[1]
    num_classes = len(le.classes_)

    # Initialize and Load Weights
    models_to_test = []
    
    # Teacher DNN
    t_dnn = TeacherDNN(input_dim, num_classes).to(device)
    t_dnn.load_state_dict(torch.load(args.tdnn_path, map_location=device)['model_state_dict'])
    models_to_test.append((t_dnn, "Teacher DNN"))

    # Teacher ResNet
    t_res = TeacherResNet(input_dim, num_classes).to(device)
    t_res.load_state_dict(torch.load(args.tres_path, map_location=device)['model_state_dict'])
    models_to_test.append((t_res, "Teacher ResNet"))

    # TGKD Student
    s_tgkd = StudentMLP(input_dim, num_classes).to(device)
    s_tgkd.load_state_dict(torch.load(args.stud_path, map_location=device)['model_state_dict'])
    models_to_test.append((s_tgkd, "Full TGKD Student"))

    # Execution
    for model, name in models_to_test:
        run_evaluation(model, test_loader, device, name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--tdnn_path', type=str, default="models/teacher_dnn_best.pth")
    parser.add_argument('--tres_path', type=str, default="models/teacher_resnet_best.pth")
    parser.add_argument('--stud_path', type=str, default="models/student_tgkd_final.pth")
    parser.add_argument('--num_parts', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1) # Low batch size for latency testing
    args = parser.parse_args()
    main(args)