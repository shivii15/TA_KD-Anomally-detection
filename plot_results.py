import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_training_results(log_path):
    # 1. Load the JSON data
    if not os.path.exists(log_path):
        print(f"❌ Error: Log file {log_path} not found.")
        return

    with open(log_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # 2. Set Academic Style
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Loss Curve ---
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e', linestyle='--', linewidth=2)
    ax1.set_title('Model Convergence (Loss)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- Panel 2: Accuracy Curve ---
    ax2.plot(epochs, history['val_acc'], label='Val Accuracy', color='#2ca02c', linewidth=2)
    ax2.set_title('Model Performance (Accuracy)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Final Formatting and Saving
    plt.tight_layout()
    output_fig = log_path.replace('.json', '.png')
    plt.savefig(output_fig, dpi=300) # 300 DPI is standard for IEEE/ACM journals
    print(f"✅ Figure saved as: {output_fig}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, required=True, help="Path to the JSON log file")
    args = parser.parse_args()
    
    plot_training_results(args.log_path)