import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest

class TrustModule(nn.Module):
    def __init__(self, method='weighted', contamination=0.05, w=[0.4, 0.4, 0.2]):
        super(TrustModule, self).__init__()
        self.method = method
        self.w = w
        self.t_base = 2.0
        self.gamma = 2.0
        
        # Your specific Anomaly Module
        self.iso_forest = IsolationForest(n_estimators=100, contamination=contamination)

    def fit_anomaly_detector(self, X_benign):
        print("🛠️ Training Isolation Forest on benign data...")
        self.iso_forest.fit(X_benign)

    def forward(self, t_logits, x_input):
        """
        Calculates t_adapt based on the selected research method.
        """
        probs = F.softmax(t_logits, dim=1)
        
        if self.method == 'weighted':
            # --- YOUR CURRENT BEST LOGIC ---
            conf, _ = torch.max(probs, dim=1)
            
            # Anomaly Score
            if_scores = self.iso_forest.score_samples(x_input.cpu().numpy())
            if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-6)
            anomaly_score = torch.FloatTensor(if_scores).to(t_logits.device)
            
            # Entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            norm_entropy = entropy / torch.log(torch.tensor(probs.size(1), dtype=torch.float))
            entropy_trust = 1.0 - norm_entropy
            
            # Trust Calculation
            trust_score = (self.w[0] * conf) + (self.w[1] * anomaly_score) + (self.w[2] * entropy_trust)
            t_adapt = self.t_base + self.gamma * (1.0 - trust_score)
            
        elif self.method == 'entropy_only':
            # --- ABLATION VARIATION 1 ---
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            trust_score = torch.exp(-entropy)
            t_adapt = self.t_base + self.gamma * (1.0 - trust_score)
            
        else:
            # --- BASELINE (Fixed T) ---
            t_adapt = torch.full((t_logits.size(0),), self.t_base).to(t_logits.device)
            trust_score = torch.ones_like(t_adapt)

        return t_adapt, trust_score