import torch
import numpy as np
from sklearn.ensemble import IsolationForest

class TrustEngine:
    def __init__(self, contamination=0.1):
        # Anomaly Module: Isolation Forest is robust for 47 IoT features
        self.anomaly_detector = IsolationForest(contamination=contamination, random_state=42)
        
    def fit_anomaly_detector(self, train_features):
        self.anomaly_detector.fit(train_features)

    def get_trust_score(self, teacher_logits, inputs, temperature=2.0):
        # 1. Confidence Module (C): Maximum Softmax Probability
        probs = torch.softmax(teacher_logits / temperature, dim=1)
        confidence, _ = torch.max(probs, dim=1)

        # 2. Anomaly Module (A): Normalized Decision Function
        # We convert sklearn scores to a [0, 1] range where 1 is "Normal"
        raw_anomaly_scores = self.anomaly_detector.decision_function(inputs.cpu().numpy())
        # Sigmoid-like normalization
        anomaly_weight = torch.tensor(1 / (1 + np.exp(-raw_anomaly_scores))).to(teacher_logits.device)

        # 3. Final Trust Score T(x)
        trust_score = confidence * anomaly_weight
        return trust_score.unsqueeze(1) # Shape [Batch, 1]