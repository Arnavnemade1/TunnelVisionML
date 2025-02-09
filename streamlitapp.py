import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import plotly.graph_objects as go

# Model definition
class QuantumTunnelingModel(nn.Module):
    def __init__(self, input_size=1):
        super(QuantumTunnelingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x).reshape(-1, 1)
            return self.forward(x)

# Data processing
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

def create_sample_data():
    num_samples = st.sidebar.slider("Number of samples", 10, 100, 50)
    noise_level = st.sidebar.slider("Noise level", 0.0, 1.0, 0.2)
    x = np.linspace(0, 10, num_samples)
    y = np.sin(x) + noise_level * np.random.randn(num_samples)
    df = pd.DataFrame({
        'ldr_value': y,
        'tunneling': (y > 0).astype(float)
    })
    return df

def main():
    st.title("Quantum Tunneling Predictor")
    st.write("""
    ### Predict quantum tunneling probability using LDR sensor data
    Upload your data or use the sample generator to test the model.
    """)
    
    model = QuantumTunnelingModel()
    processor = DataProcessor()
    
    # Sidebar
    st.sidebar.title("Controls")
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Sample Data", "Upload CSV", "Single Value"]
    )
    
    if data_source == "Sample Data":
        df = create_sample_data()
        st.write("Sample Data Preview:", df.head())
        processed_data = processor.process_data(df)
        predictions = model.predict(processed_data)
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['ldr_value'].values, name='LDR Values'))
        fig.add_trace(go.Scatter(y=predictions.numpy().flatten(), name='Predictions'))
        fig.update_layout(title='Predictions vs Actual Values')
        st.plotly_chart(fig)
        
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'ldr_value' not in df.columns:
                    st.error("CSV must contain 'ldr_value' column")
                else:
                    processed_data = processor.process_data(df)
                    predictions = model.predict(processed_data)
                    st.write("Predictions:", predictions.numpy().flatten())
            except Exception as e:
                st.error(f"Error: {str(e)}")
                
    else:  # Single Value
        value = st.number_input("Enter LDR value:", -10.0, 10.0, 0.0)
        if st.button("Predict"):
            processed_value = processor.process_data([[value]])
            prediction = model.predict(processed_value)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(prediction.item() * 100),
                title={'text': "Tunneling Probability (%)"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
