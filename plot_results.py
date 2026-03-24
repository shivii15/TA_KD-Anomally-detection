import json
import matplotlib
# ✅ Force Non-Interactive Backend (Prevents "no display" errors on servers)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import argparse
import os
import time

def plot_training_results(log_path):
    """
    Reads JSON logs and saves high-resolution plots to the logs/ directory.
    """
    history = None
    # --- Robustness: Retry logic for active training files ---
    for i in range(5):
        try:
            if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
                with open(log_path, 'r') as f:
                    history = json.load(f)
                break
        except (json.JSONDecodeError, PermissionError):
            print(f"⏳ Attempt {i+1}: Log file busy or empty, retrying in 2s...")
            time.sleep(2)

    if not history or 'train_loss' not in history:
        print(f"❌ Error: Could not read valid data from {log_path}")
        return

    # --- Setup Plotting Style ---
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    epochs = range(1, len(history['train_loss']) + 1)

    # 📉 Panel 1: Loss Convergence
    ax1.plot(epochs, history['train_loss'], label='Training Loss', color='#1f77b4', linewidth=1.5)
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', color='#ff7f0e', linestyle='--', linewidth=1.5)
    ax1.set_title('Loss Convergence (Distillation Phase)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 📈 Panel 2: Accuracy Trend
    ax2.plot(epochs, history['val_acc'], label='Validation Accuracy', color='#2ca02c', linewidth=2)
    ax2.set_title('Model Accuracy Performance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_ylim([min(history['val_acc']) - 5, 100]) # Adaptive zoom
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # --- SAVE RESULTS ---
    # Generates name like: logs/Teacher_v2.1_timestamp_plot.png
    output_image = log_path.replace('.json', '_plot.png')
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight') # 300 DPI for Journal Quality
    plt.close(fig) # Free up memory
    
    print("-" * 50)
    print(f"✅ PLOT SAVED SUCCESSFULLY")
    print(f"📂 Location: {os.path.abspath(output_image)}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Server Plotter for TGKD")
    parser.add_argument('--log_path', type=str, required=True, help="Path to the .json history file")
    args = parser.parse_args()
    
    plot_training_results(args.log_path)