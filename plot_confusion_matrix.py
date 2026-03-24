import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.metrics import confusion_matrix
from data.data_loader import preprocess_iot_data, get_dataloaders
from models.model import StudentMLP

def plot_cm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    X, y, le, _ = preprocess_iot_data(args.data_path, num_parts=args.num_parts)
    _, val_loader = get_dataloaders(X, y, batch_size=2048)
    
    # 2. Load Student Model
    model = StudentMLP(X.shape[1], len(le.classes_)).to(device)
    checkpoint = torch.load(args.student_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Collect Predictions
    all_preds = []
    all_labels = []
    print("🧪 Running evaluation on test set...")
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits, _ = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # 4. Generate Matrix
    cm = confusion_matrix(all_labels, all_preds)
    # Normalize by row (True Class) to show percentages
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 5. Plotting
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm_perc, annot=False, fmt='.1f', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    
    plt.title(f'Confusion Matrix: TGKD Student (Normalized %)\nTop Accuracy: {checkpoint.get("val_acc", "N/A"):.2f}%', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Save to logs
    output_path = args.student_path.replace(".pth", "_cm.png").replace("models/", "logs/")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion Matrix saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--student_path', type=str, required=True)
    parser.add_argument('--num_parts', type=int, default=5) # Use 5-10 parts for a clean matrix
    plot_cm(parser.parse_args())