from sklearn.ensemble import IsolationForest
import numpy as np

class TrustGate:
    def __init__(self, contamination=0.05):
        self.iso_forest = IsolationForest(
            n_estimators=100, 
            contamination=contamination, 
            random_state=42
        )
        self.t_base = 2.0  # Base temperature
        self.gamma = 2.0   # Scaling factor for low trust

    def fit(self, X_benign):
        print("Calibrating Trust Module on Golden Set...")
        self.iso_forest.fit(X_benign)

    def get_adaptive_temp(self, x_batch):
        # Convert torch batch to numpy for Scikit-Learn
        scores = self.iso_forest.score_samples(x_batch.cpu().numpy())
        
        # Normalize score to [0, 1] range (where 1 is normal)
        # score_samples returns negative values (lower is more anomalous)
        trust_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        
        # Equation: T_adapt = T_base + gamma * (1 - Trust)
        t_adapt = self.t_base + self.gamma * (1 - trust_scores)
        return torch.FloatTensor(t_adapt).to(x_batch.device)