import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from models.model import TeacherDNN
from data.data_loader import preprocess_iot_data, get_dataloaders
import torch
from sklearn.preprocessing import LabelEncoder
import sklearn

# Add this line to allow the specific sklearn class
# Allow all the common objects used in your CIC-IoT preprocessing
torch.serialization.add_safe_globals([
    np._core.multiarray._reconstruct, 
    np.ndarray, 
    np.dtype, 
    np.core.multiarray.scalar,
    sklearn.preprocessing._label.LabelEncoder,
    # Add the scaler if you saved it too
    # sklearn.preprocessing._data.StandardScaler 
])
def evaluate_teacher(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # 1. Load Test Data
    # We use a specific number of parts (e.g., the last 10) to ensure a clean test
    print(f"📊 Loading test data from {args.data_path}...")
    X_test, y_test, le, _ = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    _, test_loader = get_dataloaders(X_test, y_test, batch_size=args.batch_size)
    
    # 2. Initialize Model Architecture
    input_dim = X_test.shape[1]
    num_classes = len(le.classes_)
    model = TeacherDNN(input_dim, num_classes).to(device)
    
    # 3. Load Trained Weights
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        return

    print(f"checkpoint Loading weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle both wrapped and raw state_dicts
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    # 4. Inference Loop
    all_preds = []
    all_labels = []
    
    print("🚀 Running Inference...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            # TeacherDNN returns (logits, features) - we only need logits [0]
            logits, _ = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # 5. Generate Report
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=le.classes_, digits=4)
    
    # Save Text Report
    with open(args.output_text, "w") as f:
        f.write(f"TeacherDNN Evaluation - CIC-IoT-2023\n")
        f.write(f"Overall Accuracy: {acc*100:.4f}%\n")
        f.write("-" * 60 + "\n")
        f.write(report)
    
    print(f"\n✅ Evaluation Complete!")
    print(f"🏆 Final Accuracy: {acc*100:.2f}%")
    print(f"📄 Full report saved to: {args.output_text}")

    # 6. Plot & Save Confusion Matrix
    plt.figure(figsize=(18, 14))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=False, fmt='d', cmap='magma', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix: TeacherDNN (Acc: {acc*100:.2f}%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(args.output_img)
    print(f"🖼️ Confusion Matrix saved to: {args.output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TeacherDNN Performance")
    
    # Paths
    parser.add_argument('--model_path', type=str, default='models/teacher_dnn_best.pth',
                        help='Path to the trained .pth file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CIC-IoT-2023 dataset folder')
    
    # Settings
    parser.add_argument('--num_parts', type=int, default=10,
                        help='Number of data parts to use for testing')
    parser.add_argument('--batch_size', type=int, default=1024)
    
    # Outputs
    parser.add_argument('--output_text', type=str, default='teacher_dnn_report.txt')
    parser.add_argument('--output_img', type=str, default='teacher_dnn_cm.png')

    args = parser.parse_args()
    evaluate_teacher(args)