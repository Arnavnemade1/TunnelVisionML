import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def process_data(self, data):
        if isinstance(data, pd.DataFrame):
            values = data['ldr_value'].values.reshape(-1, 1)
        else:
            values = np.array(data).reshape(-1, 1)
        
        scaled_data = self.scaler.fit_transform(values)
        return torch.FloatTensor(scaled_data)
