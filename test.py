import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# Import custom modules
from data.data_loader import preprocess_iot_data
from models.model import StudentMLP

def run_test(model_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Starting Evaluation on: {device}")

    # 1. Load Checkpoint
    # We use weights_only=False because the checkpoint contains the LabelEncoder
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    run_id = checkpoint.get('run_id', 'unknown_run')
    le = checkpoint['le']
    
    # 2. Prepare Data
    print("📊 Loading Test Data...")
    _, X_test, _, y_test, _, _ = preprocess_iot_data(data_path)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # 3. Initialize & Load Model
    input_dim = X_test.shape[1]
    num_classes = len(le.classes_)
    model = StudentMLP(input_dim, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Inference
    print("🚀 Running Inference...")
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

    # 5. Generate Metrics
    report = classification_report(y_test, preds, target_names=le.classes_)
    print("\n📝 Classification Report:\n", report)

    # 6. Plot Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix: {run_id}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save Plot
    plot_path = model_path.replace(".pth", "_cm.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"✅ Confusion Matrix saved to: {plot_path}")

if __name__ == "__main__":
    # Update these paths to point to your specific best model and data
    MODEL_FILE = "./outputs/TGKD_small_a0.5_b0.1_20260315_1430/best_student_TGKD_small_a0.5_b0.1_20260315_1430.pth"
    DATA_FILE = "./data/CICIoT2023_xxsmall.csv"
    
    run_test(MODEL_FILE, DATA_FILE)