import torch.nn as nn

class TeacherDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TeacherDNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128) # Intermediate layer for alignment
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.feature_extractor(x)
        logits = self.classifier(feat)
        return logits, feat

class StudentMLP(nn.Module):
    def __init__(self, input_dim, num_classes, size='small'):
        super(StudentMLP, self).__init__()
        # Variation: Tiny vs Small
        hidden = 32 if size == 'tiny' else 64
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32) # Intermediate layer for alignment
        )
        self.classifier = nn.Linear(32, num_classes)
        
        # PROJECTION LAYER: Maps Student (32) to Teacher (128)
        self.projection = nn.Linear(32, 128)

    def forward(self, x):
        feat = self.feature_extractor(x)
        projected_feat = self.projection(feat) # Alignment step
        logits = self.classifier(feat)
        return logits, projected_feat
    
# --- ADD THE NEW RESEARCH MODEL ---
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim), # Added for stability in 169-file runs
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
        return self.output_layer(self.res_stack(self.input_layer(x))) 