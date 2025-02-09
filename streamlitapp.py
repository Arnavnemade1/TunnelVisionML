import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from model import QuantumTunnelingModel
from data_processor import DataProcessor
import torch

def create_sample_data():
    st.sidebar.markdown("### Sample Data Generator")
    num_samples = st.sidebar.slider("Number of samples", 10, 100, 50)
    noise_level = st.sidebar.slider("Noise level", 0.0, 1.0, 0.2)
    
    # Generate sample data
    x = np.linspace(0, 10, num_samples)
    y = np.sin(x) + noise_level * np.random.randn(num_samples)
    df = pd.DataFrame({
        'ldr_value': y,
        'tunneling': (y > 0).astype(float)
    })
    return df

def plot_predictions(original_values, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=original_values,
        name='Original LDR Values',
        mode='lines+markers'
    ))
    fig.add_trace(go.Scatter(
        y=predictions,
        name='Tunneling Probability',
        mode='lines+markers'
    ))
    fig.update_layout(
        title='Quantum Tunneling Predictions',
        xaxis_title='Sample',
        yaxis_title='Value',
        height=500
    )
    return fig

def main():
    st.title("Quantum Tunneling Predictor")
    st.markdown("""
    This application predicts quantum tunneling probability based on LDR (Light Dependent Resistor) sensor values.
    Upload your data or use the sample data generator to test the model.
    """)
    
    # Initialize model and processor
    model = QuantumTunnelingModel()
    processor = DataProcessor()
    
    # Sidebar for data source selection
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Upload CSV", "Generate Sample Data", "Single Value Prediction"]
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your data CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Data Preview:", df.head())
                processed_data = processor.process_data(df)
                predictions = model.predict(processed_data)
                
                # Plot results
                fig = plot_predictions(df['ldr_value'].values, predictions.numpy().flatten())
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    elif data_source == "Generate Sample Data":
        df = create_sample_data()
        st.write("Generated Data Preview:", df.head())
        processed_data = processor.process_data(df)
        predictions = model.predict(processed_data)
        
        # Plot results
        fig = plot_predictions(df['ldr_value'].values, predictions.numpy().flatten())
        st.plotly_chart(fig)
    
    else:  # Single Value Prediction
        st.header("Single Value Prediction")
        value = st.number_input("Enter LDR sensor value:", value=0.0)
        if st.button("Predict"):
            processed_value = processor.process_data([[value]])
            prediction = model.predict(processed_value)
            st.write(f"Tunneling Probability: {prediction.item():.2%}")
            
            # Show gauge chart for single prediction
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = float(prediction.item() * 100),
                title = {'text': "Tunneling Probability"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ]
                }
            ))
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()

# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import QuantumTunnelingModel
from data_processor import DataProcessor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def train_model(data_path, epochs=100, batch_size=32):
    # Load and process data
    df = pd.read_csv(data_path)
    processor = DataProcessor()
    X = processor.process_data(df['ldr_value'])
    y = torch.FloatTensor(df['tunneling'].values).reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Create model
    model = QuantumTunnelingModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
    
    return model
