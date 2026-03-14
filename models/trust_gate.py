import torch
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import IsolationForest

class TGKD_TrustModule:
    def __init__(self, contamination=0.05, w=[0.4, 0.4, 0.2]):
        """
        w: Weights for [Confidence C(x), Anomaly A(x), Entropy (1-H(x))]
        """
        self.iso_forest = IsolationForest(n_estimators=100, contamination=contamination)
        self.w = w
        self.t_base = 2.0  # Base temperature
        self.gamma = 2.0   # Scaling factor

    def fit_anomaly_detector(self, X_benign):
        print("Training Anomaly Module (A) on benign data...")
        self.iso_forest.fit(X_benign)

    def calculate_trust(self, teacher_logits, x_input):
        probs = F.softmax(teacher_logits, dim=1)
        
        # 1. Confidence C(x): Max Softmax Probability
        conf, _ = torch.max(probs, dim=1)
        
        # 2. Anomaly Score A(x): Isolation Forest (scaled to [0,1])
        # score_samples returns negative values (lower is more anomalous)
        if_scores = self.iso_forest.score_samples(x_input.cpu().numpy())
        if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-6)
        anomaly_score = torch.FloatTensor(if_scores).to(teacher_logits.device)
        
        # 3. Entropy-based Trust (1 - H(x)): High entropy = low trust
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        norm_entropy = entropy / torch.log(torch.tensor(probs.size(1), dtype=torch.float))
        entropy_trust = 1.0 - norm_entropy
        
        # Final Trust Score Equation from image: T(x) = w1*C + w2*A + w3*(1-H)
        trust_score = (self.w[0] * conf) + (self.w[1] * anomaly_score) + (self.w[2] * entropy_trust)
        
        # Map Trust Score to Adaptive Temperature: T_adapt = T_base + gamma*(1 - Trust)
        t_adapt = self.t_base + self.gamma * (1.0 - trust_score)
        
        return t_adapt, trust_score
    