import torch.nn as nn

class TGKD_Trainer:
    def __init__(self, student, teacher, trust_module, alpha=0.5, beta=0.1):
        self.student = student
        self.teacher = teacher
        self.trust_module = trust_module
        self.alpha = alpha   # Weight for Distillation
        self.beta = beta     # Weight for Feature Alignment
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kd = nn.KLDivLoss(reduction='none') # 'none' to apply sample-wise T
        self.criterion_feat = nn.MSELoss()

    def compute_loss(self, x, labels):
        # Forward pass
        with torch.no_grad():
            t_logits, t_feat = self.teacher(x)
            t_adapt, _ = self.trust_module.calculate_trust(t_logits, x)
        
        s_logits, s_feat = self.student(x)

        # 1. Task Loss (L_CE)
        l_ce = self.criterion_ce(s_logits, labels)

        # 2. Gated Distillation Loss (L_KD)
        # Reshape t_adapt for broadcasting [batch_size, 1]
        t_adapt = t_adapt.unsqueeze(1)
        
        soft_targets = F.softmax(t_logits / t_adapt, dim=1)
        soft_log_probs = F.log_softmax(s_logits / t_adapt, dim=1)
        
        # Pointwise KL then mean
        l_kd = self.criterion_kd(soft_log_probs, soft_targets).sum(dim=1)
        l_kd = (l_kd * (t_adapt.squeeze()**2)).mean()

        # 3. Feature Alignment Loss (L_Feat)
        l_feat = self.criterion_feat(s_feat, t_feat)

        # Final Combined Loss
        total_loss = (1 - self.alpha) * l_ce + (self.alpha * l_kd) + (self.beta * l_feat)
        
        return total_loss