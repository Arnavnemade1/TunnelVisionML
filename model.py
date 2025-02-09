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
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

# 3. data_processor.py (Data Processing)
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def process_sensor_data(self, raw_data):
        """Process raw LDR sensor data"""
        # Convert to DataFrame if not already
        if isinstance(raw_data, list):
            df = pd.DataFrame(raw_data, columns=['ldr_value'])
        else:
            df = pd.DataFrame(raw_data)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(df)
        return torch.FloatTensor(scaled_data)
    
    def load_bluetooth_data(self, data_string):
        """Process data received from HC-05"""
        # Split the data string and convert to float
        values = [float(x) for x in data_string.split(',')]
        return values
