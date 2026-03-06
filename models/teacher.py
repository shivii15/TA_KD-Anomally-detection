import torch
import torch.nn as nn

class IoTTeacher(nn.Module):
    """
    High-capacity Teacher Network for 47-feature IoT Tabular Data.
    Designed with Dropout and Batch Normalization for stability and 
    wide hidden layers to act as a robust knowledge source.
    """
    def __init__(self, input_dim=47, num_classes=34):
        super(IoTTeacher, self).__init__()
        
        self.features = nn.Sequential(
            # Layer 1: Expansion
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2: Deep Processing
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3: Feature Refinement
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Layer 4: Bottleneck before Logits
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Final Output Layer (Logits)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Returns raw logits. 
        Note: Softmax is applied in the Distillation Loss, not here.
        """
        x = self.features(x)
        return self.classifier(x)

def get_teacher_model(device='cpu'):
    model = IoTTeacher()
    return model.to(device)

# Example usage for verification:
if __name__ == "__main__":
    test_input = torch.randn(8, 47) # Batch of 8, 47 features
    model = IoTTeacher()
    output = model(test_input)
    print(f"Teacher Output Shape: {output.shape}") # Should be [8, 34]