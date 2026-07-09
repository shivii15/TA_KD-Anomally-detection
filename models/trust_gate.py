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
        confidence_weight=0.4,
        anomaly_weight=0.3,
        entropy_weight=0.2,
        disagreement_weight=0.1,
        base_temperature=2.0,
        lambda_temp=1.0,
        trust_method="weighted"
    ):
        super().__init__()

        # -------------------------------
        # Trust Weights
        # -------------------------------
        self.confidence_weight = confidence_weight
        self.anomaly_weight = anomaly_weight
        self.entropy_weight = entropy_weight
        self.disagreement_weight = disagreement_weight

        # -------------------------------
        # Temperature Parameters
        # -------------------------------
        self.base_temperature = base_temperature
        self.lambda_temp = lambda_temp

        # -------------------------------
        # Trust Strategy
        # -------------------------------
        self.trust_method = trust_method

        # -------------------------------
        # Placeholder for trained
        # Isolation Forest
        # -------------------------------
        self.anomaly_detector = None


    def load_anomaly_detector(self, model_path):

        package = joblib.load(model_path)

        self.anomaly_detector = package["model"]

        self.scaler = package["scaler"]

        self.dataset = package["dataset"]

        self.feature_dim = package["feature_dim"]

        self.classes = package["classes"]

        print("✅ Isolation Forest loaded successfully.")
    
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
    
    def compute_anomaly_score(self, x_input):

        if self.anomaly_detector is None:

            raise RuntimeError(
                "Anomaly detector has not been loaded."
            )

        scores = self.anomaly_detector.score_samples(
            x_input.detach().cpu().numpy()
        )

        scores = torch.tensor(
            scores,
            dtype=torch.float32,
            device=x_input.device
        )

        # Normalize to [0,1]

        scores = (scores - scores.min()) / (
            scores.max() - scores.min() + 1e-8
        )

        return scores
    
    def compute_disagreement(
        self,
        teacher_logits,
        student_logits
    ):
        """
        Compute teacher-student disagreement.

        D(x) = ||P_teacher - P_student||₂
        """

        teacher_probs = F.softmax(
            teacher_logits,
            dim=1
        )

        student_probs = F.softmax(
            student_logits,
            dim=1
        )

        disagreement = torch.norm(
            teacher_probs - student_probs,
            p=2,
            dim=1
        )

        return disagreement
        
    def compute_trust_score(
        self,
        confidence,
        entropy_trust,
        anomaly_score,
        disagreement
    ):
        """
        Compute the final CRED trust score.

        T(x) =
            α * Confidence
        + β * EntropyTrust
        + γ * Reliability
        - δ * Disagreement
        """

        reliability = 1.0 - anomaly_score

        if self.trust_method == "weighted":

            trust_score = (
                self.confidence_weight * confidence
                + self.entropy_weight * entropy_trust
                + self.anomaly_weight * reliability
                - self.disagreement_weight * disagreement
            )

        elif self.trust_method == "confidence_only":

            trust_score = confidence

        elif self.trust_method == "entropy_only":

            trust_score = entropy_trust

        elif self.trust_method == "anomaly_only":

            trust_score = reliability

        elif self.trust_method == "fixed":

            trust_score = torch.ones_like(confidence)

        else:

            raise ValueError(
                f"Unknown trust method: {self.trust_method}"
            )

        trust_score = torch.clamp(
            trust_score,
            min=0.0,
            max=1.0
        )

        return trust_score

    def compute_temperature(
        self,
        trust_score
    ):
        """
        Compute adaptive temperature from trust score.
        """

        temperature = (
            self.base_temperature
            * (
                1.0
                + self.lambda_temp
                * (1.0 - trust_score)
            )
        )

        return temperature
    
    def forward(
        self,
        teacher_logits,
        student_logits,
        x_input
    ):
        """
        Compute all trust-related quantities required by TGKD.
        """

        confidence = self.compute_confidence(
            teacher_logits
        )

        entropy_trust = self.compute_entropy(
            teacher_logits
        )

        anomaly_score = self.compute_anomaly_score(
            x_input
        )

        disagreement = self.compute_disagreement(
            teacher_logits,
            student_logits
        )

        trust_score = self.compute_trust_score(
            confidence,
            entropy_trust,
            anomaly_score,
            disagreement
        )

        temperature = self.compute_temperature(
            trust_score
        )

        return {

            "confidence": confidence,

            "entropy_trust": entropy_trust,

            "anomaly_score": anomaly_score,

            "disagreement": disagreement,

            "trust_score": trust_score,

            "temperature": temperature

        }