import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
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
        return self.relu(x + self.fc(x))

class TeacherResNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())
        self.res_stack = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            ResidualBlock(256)
        )
        self.output_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        # 1. Extract intermediate features (256-dim)
        features = self.res_stack(self.input_layer(x))
        
        # 2. Extract logits for classification
        logits = self.output_layer(features)
        
        # Returns BOTH for Hybrid Distillation
        return logits, features

class StudentMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        
        # --- PDF METHODOLOGY ADDITION ---
        # Projects 64-dim student features to 256-dim to match Teacher's ResNet
        self.projector = nn.Linear(64, 256) 
        
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x1 = self.layer1(x)
        # Internal features (64-dim)
        feat_internal = self.layer2(x1)
        
        # 1. Projected features for MSE loss with Teacher (L_feat)
        proj_features = self.projector(feat_internal)
        
        # 2. Logits for classification
        logits = self.classifier(feat_internal)
        
        return logits, proj_features

class TeacherDNN(nn.Module):
    """Standard DNN for V1 Baseline comparisons"""
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