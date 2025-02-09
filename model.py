import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QuantumTunnelingModel(nn.Module):
    def __init__(self, input_size=1):
        super(QuantumTunnelingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x).reshape(-1, 1)
            return self.forward(x)
