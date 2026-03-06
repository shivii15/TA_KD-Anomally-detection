import torch
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

class TrustModule:
    """
    Trust-Gated Logic: T(x) = C(x) * A(x)
    C(x): Teacher Confidence (Maximum Softmax Probability)
    A(x): Anomaly Score (Isolation Forest Decision Function)
    """
    def __init__(self, contamination=0.05):
        # We use Isolation Forest as the Anomaly Module (A) 
        # because it is highly effective for 47D tabular IoT features.
        self.anomaly_module = IsolationForest(
            n_estimators=100, 
            contamination=contamination, 
            random_state=42,
            n_jobs=-1
        )
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def fit_anomaly_detector(self, normal_traffic_features):
        """
        Trains the Anomaly Module on 'Normal' or 'Trusted' traffic 
        to establish a baseline for what 'Trustworthy' data looks like.
        """
        print("🔧 Fitting Anomaly Module on training features...")
        # Isolation Forest expects a 2D numpy array
        features_np = normal_traffic_features.cpu().numpy()
        self.anomaly_module.fit(features_np)
        
        # We pre-calculate scores to calibrate our [0, 1] scaler
        raw_scores = self.anomaly_module.decision_function(features_np).reshape(-1, 1)
        self.scaler.fit(raw_scores)
        
        self.is_fitted = True
        print("✅ Trust Module Calibrated.")

    def get_trust_score(self, teacher_logits, inputs, tau=2.0):
        """
        Calculates the per-sample Trust Score T(x).
        teacher_logits: [Batch, 34]
        inputs: [Batch, 47]
        """
        if not self.is_fitted:
            raise RuntimeError("TrustModule must be fitted on data before inference.")

        # 1. Calculate Confidence C(x) 
        # We use Temperature Scaling (tau) to get more nuanced probabilities
        probs = torch.softmax(teacher_logits / tau, dim=1)
        confidence, _ = torch.max