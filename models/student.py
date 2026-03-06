import torch
import torch.nn as nn

class IoTStudent(nn.Module):
    def __init__(self, input_dim=47, num_classes=34):
        super(IoTStudent, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.encoder(x)

class IoTTeacher(nn.Module):
    def __init__(self, input_dim=47, num_classes=34):
        super(IoTTeacher, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)