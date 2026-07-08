from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
# At the top of train.py, add:
from contextlib import nullcontext

class TGKD_Trainer:
    def __init__(self, student, teacher, trust_module, alpha=0.5, beta=0.1, accumulation_steps=1):
        self.student = student
        self.teacher = teacher
        self.trust_module = trust_module
        self.alpha = alpha
        self.beta = beta
        self.accumulation_steps = accumulation_steps
        
        # Loss Functions
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kd = nn.KLDivLoss(reduction='batchmean')
        self.criterion_feat = nn.MSELoss() 

        # Modern Scaler for T4 GPU
        self.scaler = torch.amp.GradScaler('cuda')

    def train_epoch(self, loader, optimizer, device):
        # Silence sklearn warnings once at start of epoch
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

        self.student.train()
        self.teacher.eval() 
        
        total_epoch_loss = 0
        total_packets = 0
        optimizer.zero_grad()
        
        # Start the clock
        start_time = time.time()

        # Wrap loader in tqdm for a clean progress bar
        #pbar = tqdm(loader, desc="Training Batches", leave=False)
        pbar = tqdm(loader, desc="Training", disable=True)

        self.print_gpu_memory(device)
        
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            total_packets += x.size(0)

            # 1. Forward pass with modern autocasting
            context = torch.amp.autocast('cuda') if device.type == 'cuda' else nullcontext()

            with context:
                with torch.no_grad():
                    t_logits, t_feat = self.teacher(x)
                    # Get trust weight (A)
                    #t_adapt, _ = self.trust_module(t_logits, x) # Calling the module directly triggers forward()
                    #t_adapt, _ = self.trust_module.calculate_trust(t_logits, x)
                
                s_logits, s_feat = self.student(x)

                trust_outputs = self.trust_module(
                teacher_logits=t_logits,
                student_logits=s_logits,
                x_input=x
                )

                if i == 0:

                    print("\n===============Trust Module Output=================")

                    print("\nTrust Statistics")

                    print(f"Confidence   : {confidence.mean():.4f}")

                    print(f"Entropy      : {entropy.mean():.4f}")

                    print(f"Anomaly      : {anomaly.mean():.4f}")

                    print(f"Disagreement : {disagreement.mean():.4f}")

                    print(f"Trust Score  : {trust_score.mean():.4f}")

                    print(f"Temperature  : {temperature.mean():.4f}")
                
                    print("==================================\n")


                temperature = trust_outputs["temperature"]
                trust_score = trust_outputs["trust_score"]
                confidence = trust_outputs["confidence"]
                entropy = trust_outputs["entropy_trust"]
                anomaly = trust_outputs["anomaly_score"]
                disagreement = trust_outputs["disagreement"]
                

                # --- LOSS CALCULATION ---
                l_ce = self.criterion_ce(s_logits, y)
                
                # Temperature Scaling with Trust Module
                temperature_batch = temperature.unsqueeze(1) 
                soft_targets = F.softmax(
                    t_logits / temperature_batch,
                    dim=1
                ) 
                soft_log_probs = F.log_softmax(
                    s_logits / temperature_batch,
                    dim=1
                )                
                # KD Loss (Trust-Gated)
                l_kd = self.criterion_kd(soft_log_probs, soft_targets)
                l_kd = (
                    l_kd *
                    (temperature ** 2)
                ).mean()                
                # Feature Alignment Loss
                l_feat = self.criterion_feat(s_feat, t_feat)
                
                # Total Combined Loss
                loss = ((1 - self.alpha) * l_ce + (self.alpha * l_kd) + (self.beta * l_feat))
                loss = loss / self.accumulation_steps

            # 2. Backpropagation
            self.scaler.scale(loss).backward()

            # 3. Accumulation Step
            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            # Update progress bar info
            current_loss = loss.item() * self.accumulation_steps
            total_epoch_loss += current_loss
            pbar.set_postfix({"batch_loss": f"{current_loss:.4f}"})

            # Calculate duration and print performance
            #epoch_duration = time.time() - start_time
            #self.print_performance(device, total_packets, epoch_duration)

        return total_epoch_loss / len(loader)
    
    def evaluate(self, loader, device, epoch=None, total_epochs=None, label_names=None):
        self.student.eval()
        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Evaluating", leave=False):
                x = x.to(device)
                logits, _ = self.student(x)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
            # Calculate Metrics
            acc = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )
            
            # Generate Confusion Matrix on the last epoch
            if epoch == total_epochs:
                cm = confusion_matrix(all_labels, all_preds)
                plt.figure(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=label_names, yticklabels=label_names)
                plt.title(f'Confusion Matrix - Epoch {epoch}')
                plt.ylabel('Actual Category')
                plt.xlabel('Predicted Category')
                plt.savefig(f'confusion_matrix_epoch_{epoch}.png')
                print(f"✅ Confusion Matrix saved as 'confusion_matrix_epoch_{epoch}.png'")
                plt.show()

        return {
                "accuracy": acc, "precision": precision, "recall": recall, "f1": f1, 'timestamp': datetime.now().strftime("%H:%M:%S")
            }
    
    def print_performance(self, device, total_packets, duration):
            """Prints GPU memory usage and processing throughput."""
            throughput = total_packets / duration
            print(f"\n🚀 Performance Metrics:")
            print(f"   - Throughput: {throughput:.2f} Packets/Second")
            
            if device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"   - GPU VRAM: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")

    def print_gpu_memory(self, device):
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"📊 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
