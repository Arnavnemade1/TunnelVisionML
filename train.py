import torch
import torch.nn as nn
import torch.optim as optim
from model import QuantumTunnelingModel
from data_processor import DataProcessor

def train_model(data_path, epochs=100):
    # Load and process data
    processor = DataProcessor()
    data = processor.process_data(data_path)
    
    # Create model
    model = QuantumTunnelingModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model
