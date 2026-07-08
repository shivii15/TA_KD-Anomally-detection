import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F


class TGKDTrustModule(nn.Module):
    """
    Trust Gate Module for Trust-Gated Knowledge Distillation (TGKD).

    Responsibilities:
        1. Compute teacher confidence.
        2. Compute prediction entropy.
        3. Compute anomaly score.
        4. Compute trust score.
        5. Compute adaptive temperature.
    """

    def __init__(
        self,
        trust_method="weighted",
        confidence_weight=0.4,
        anomaly_weight=0.4,
        entropy_weight=0.2,
        base_temperature=2.0,
        gamma=2.0,
    ):
        super().__init__()

        self.trust_method = trust_method

        self.confidence_weight = confidence_weight
        self.anomaly_weight = anomaly_weight
        self.entropy_weight = entropy_weight

        self.base_temperature = base_temperature
        self.gamma = gamma

        self.anomaly_detector = None

    def load_anomaly_detector(self, detector_path):
        """
        Load a pretrained anomaly detector.
        """

        detector = joblib.load(detector_path)

        if isinstance(detector, dict):
            detector = detector["model"]

        self.anomaly_detector = detector

        print("✅ Anomaly detector loaded successfully.")
    
    def compute_confidence(self, teacher_logits):

        probabilities = F.softmax(teacher_logits, dim=1)

        confidence, _ = torch.max(probabilities, dim=1)

        return confidence
    
    def compute_entropy(self, teacher_logits):

        probabilities = F.softmax(teacher_logits, dim=1)

        entropy = -torch.sum(
            probabilities * torch.log(probabilities + 1e-10),
            dim=1,
        )

        entropy = entropy / torch.log(
            torch.tensor(
                probabilities.size(1),
                device=teacher_logits.device,
                dtype=torch.float32,
            )
        )

        entropy_trust = 1.0 - entropy

        return entropy_trust