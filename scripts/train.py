import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

class TGKD_Trainer:
    def __init__(self, student, teacher, trust_module, alpha=0.5, beta=0.1, accumulation_steps=1):
        self.student = student
        self.teacher = teacher
        self.trust_module = trust_module
        self.alpha = alpha
        self.beta = beta
        self.accumulation_steps = accumulation_steps
        
        # --- ADD THESE LINES ---
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kd = nn.KLDivLoss(reduction='batchmean')
        self.criterion_mse = nn.MSELoss() 
        self.criterion_feat = nn.MSELoss()  # <--- ADD THIS LINE
        # -----------------------

        # Update the Scaler for the new PyTorch version
        self.scaler = torch.amp.GradScaler('cuda')

    def train_epoch(self, loader, optimizer, device):
        self.student.train()
        self.teacher.eval() # Teacher stays frozen
        
        optimizer.zero_grad()
        
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            # 1. Runs the forward pass with autocasting
            with autocast():
                # Get Teacher outputs and Trust metrics without calculating gradients
                with torch.no_grad():
                    t_logits, t_feat = self.teacher(x)
                    t_adapt, _ = self.trust_module.calculate_trust(t_logits, x)
                
                # Student forward pass
                s_logits, s_feat = self.student(x)
                
                # --- LOSS CALCULATION ---
                # Task Loss
                l_ce = self.criterion_ce(s_logits, y)
                
                # Gated Distillation Loss
                t_adapt = t_adapt.unsqueeze(1)
                soft_targets = F.softmax(t_logits / t_adapt, dim=1)
                soft_log_probs = F.log_softmax(s_logits / t_adapt, dim=1)
                #l_kd = self.criterion_kd(soft_log_probs, soft_targets).sum(dim=1)
                l_kd = self.criterion_kd(soft_log_probs, soft_targets)
                l_kd = (l_kd * (t_adapt.squeeze()**2)).mean()
                
                # Feature Alignment Loss
                l_feat = self.criterion_feat(s_feat, t_feat)
                
                # Total Combined Loss (scaled by accumulation steps)
                total_loss = ((1 - self.alpha) * l_ce + (self.alpha * l_kd) + (self.beta * l_feat))
                total_loss = total_loss / self.accumulation_steps

            # 2. Backpropagation with Scaled Gradients
            self.scaler.scale(total_loss).backward()

            # 3. Step Optimizer only after enough gradients have accumulated
            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                
                # Clean up cache for small GPUs
                torch.cuda.empty_cache()

        return total_loss.item() * self.accumulation_steps