import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. SHARED COMPONENTS ---

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
        return self.relu(x + self.fc(x))

# --- 2. TEACHER MODELS (The Committee of Experts) ---

class TeacherResNet(nn.Module):
    """High-Capacity Teacher (v2.1): Spatial Expert for CIC-IoT-2023."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        self.res_stack = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            ResidualBlock(256) 
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.res_stack(self.input_layer(x))
        logits = self.classifier(features)
        return logits, features

class TeacherTransformer(nn.Module):
    """Attention-Based Teacher: Relational Expert for Feature Correlations."""
    def __init__(self, input_dim, num_classes, embed_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Standard Transformer Encoder for Tabular Feature Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # [batch, features] -> [batch, 1, embed_dim]
        x_emb = self.input_proj(x).unsqueeze(1)
        features = self.transformer(x_emb).squeeze(1)
        logits = self.classifier(features)
        return logits, features

class TeacherLSTM(nn.Module):
    """Recurrent Teacher: Sequential Expert for Temporal Attack Patterns."""
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        # Bi-LSTM to capture patterns in both directions of the feature vector
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True
        )
        # Bidirectional hidden_dim (128*2) = 256, matching ResNet/Transformer feature space
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # [batch, features] -> [batch, 1, features]
        x_seq = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x_seq)
        # Concatenate final hidden states from forward and backward passes
        features = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        logits = self.classifier(features)
        return logits, features

# --- 3. BASELINE MODELS ---

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

# --- 4. STUDENT MODEL ---

class StudentMLP(nn.Module):
    """Lightweight Student with Projection-based Feature Alignment."""
    def __init__(self, input_dim, num_classes, teacher_feat_dim=256):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        
        # Alignment Projector: Maps 64-dim student features to 256-dim Teacher space
        self.projector = nn.Linear(64, teacher_feat_dim) 
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        f1 = self.layer1(x)
        feat_internal = self.layer2(f1)
        proj_features = self.projector(feat_internal)
        logits = self.classifier(feat_internal)
        return logits, proj_features