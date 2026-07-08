import torch
import torch.nn as nn

class TeacherTransformer(nn.Module):
    """
    Transformer-based Teacher
    Captures global feature relationships using self-attention.
    Returns:
        logits   : Classification output
        features : 256-dimensional representation for feature distillation
    """

    def __init__(self, input_dim, num_classes, embed_dim=256):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        # (batch, features)
        x = self.embedding(x)

        # (batch, 1, embed_dim)
        x = x.unsqueeze(1)

        # Self-attention
        features = self.transformer(x)

        # (batch, embed_dim)
        features = features.squeeze(1)

        logits = self.classifier(features)

        return logits, features