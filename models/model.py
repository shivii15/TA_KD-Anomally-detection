import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Refined Residual Block with Dropout for better Teacher generalization."""
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # Skip connection to preserve gradient flow in deep IoT feature spaces
        return self.relu(x + self.fc(x))

class TeacherResNet(nn.Module):
    """High-Capacity Teacher for CIC-IoT-2023 Distillation."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        # Deep stack to capture complex attack signatures
        self.res_stack = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            ResidualBlock(256) # Final 256-dim feature space
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # Extract 256-dim features for L_feat alignment
        features = self.res_stack(self.input_layer(x))
        logits = self.classifier(features)
        return logits, features

class StudentMLP(nn.Module):
    """Lightweight Student with Projection-based Feature Alignment."""
    def __init__(self, input_dim, num_classes, teacher_feat_dim=256):
        super().__init__()
        # Compact hidden layers for low-latency IoT inference
        self.layer1 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        
        # --- ACADEMIC NOVELTY: The Alignment Projector ---
        # Maps 64-dim student features to 256-dim Teacher space for MSE Loss
        self.projector = nn.Linear(64, teacher_feat_dim) 
        
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        f1 = self.layer1(x)
        feat_internal = self.layer2(f1)
        
        # 1. Projected features (Used only during training for L_feat)
        # During pure inference on IoT, this branch can be ignored to save cycles
        proj_features = self.projector(feat_internal)
        
        # 2. Logits for final classification
        logits = self.classifier(feat_internal)
        
        return logits, proj_features

class TeacherDNN(nn.Module):
    """Standard DNN for V1 Baseline comparisons in your paper."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits, feat