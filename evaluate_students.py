import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from models.model import StudentMLP  # Ensure this matches your student arch
from data.data_loader import preprocess_iot_data

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data (Use 5-10 parts for a statistically significant result)
    print("📂 Loading Test Data...")
    X_test, y_test, le, scaler = preprocess_iot_data("/root/.cache/kagglehub/datasets/akashdogra/cic-iot-2023/versions/1", num_parts=5)
    
    # 2. Load Student Model
    input_size = X_test.shape[1]
    num_classes = len(le.classes_)
    student = StudentMLP(input_size, num_classes).to(device)
    
    # Load your best weights
    checkpoint = torch.load("models/student_tgkd_best.pth", map_location=device)
    student.load_state_dict(checkpoint['model_state_dict'])
    student.eval()

    # 3. Inference
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = student(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
    
    y_pred = predicted.cpu().numpy()

    # 4. Generate Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    
    print(f"\n✨ Student Accuracy: {acc*100:.2f}%")
    print("\n📝 Classification Report:\n", report)

    # Save report to text for Arpita Ma'am to email
    with open("results/student_final_report.txt", "w") as f:
        f.write(f"Student Accuracy: {acc*100:.2f}%\n")
        f.write(report)

    # 5. Confusion Matrix Visualization
    plt.figure(figsize=(15, 12))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix - TG-MKD Student")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("results/student_confusion_matrix.png")
    print("📊 Confusion Matrix saved to results/student_confusion_matrix.png")

if __name__ == "__main__":
    evaluate()