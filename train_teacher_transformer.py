import torch
import torch.nn as nn

class TeacherTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TeacherTransformer, self). __init__()
        # Project 1D features into a higher-dimensional embedding space
        self.embedding = nn.Linear(input_dim, 128)
        
        # Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, 
            nhead=8, 
            dim_feedforward=256, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, features) -> transform to (batch, 1, 128)
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1) # Back to (batch, 128)
        return self.fc(x)