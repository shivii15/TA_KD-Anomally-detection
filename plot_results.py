import json
import matplotlib.pyplot as plt
import argparse
import os
import time

def plot_training_results(log_path):
    # --- SOTA Robustness: Retry logic for active training ---
    history = None
    for _ in range(5):  # Try 5 times
        try:
            if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
                with open(log_path, 'r') as f:
                    history = json.load(f)
                break 
        except json.JSONDecodeError:
            print("⏳ Log file is being updated... retrying in 1s")
            time.sleep(1)
    
    if not history:
        print(f"❌ Error: Could not read valid JSON from {log_path}. Is Epoch 1 finished?")
        return

    # --- Rest of your plotting code ---
    epochs = range(1, len(history['train_loss']) + 1)
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e', linestyle='--')
    ax1.set_title('Loss Convergence')
    ax1.legend()

    ax2.plot(epochs, history['val_acc'], label='Val Accuracy', color='#2ca02c')
    ax2.set_title('Accuracy Trend')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(log_path.replace)