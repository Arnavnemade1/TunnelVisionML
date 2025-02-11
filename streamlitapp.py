import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go
import math
import random
import time
from io import StringIO

class QuantumTunnelingPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def simulate_tunneling(self, ldr_value):
        # Enhanced simulation with more realistic quantum behavior
        base_probability = (math.sin(ldr_value) + 1) / 2
        noise = random.uniform(-0.1, 0.1)
        potential_barrier = math.exp(-ldr_value / 5)  # Simulate potential barrier effect
        probability = max(0, min(1, (base_probability + noise) * potential_barrier))
        return probability
    
    def prepare_data(self, ldr_values):
        # Create features from LDR values
        features = pd.DataFrame({
            'ldr_value': ldr_values,
            'potential_barrier': np.exp(-np.array(ldr_values) / 5),
            'wave_component': np.sin(np.array(ldr_values))
        })
        return features
    
    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model needs to be trained first!")
        return self.model.predict(X), self.model.predict_proba(X)

def main():
    st.title("🌌 Quantum Tunneling ML Predictor")
    st.write("""
    ### Advanced Quantum Tunneling Analysis with Machine Learning
    This application combines real LDR measurements with machine learning to predict quantum tunneling probability.
    """)
    
    predictor = QuantumTunnelingPredictor()
    
    # Sidebar for mode selection
    mode = st.sidebar.radio("Select Mode", ["Upload Data", "Single Prediction", "Real-time Simulation"])
    
    if mode == "Upload Data":
        st.write("### 📤 Upload Your Tunneling Data")
        uploaded_file = st.file_uploader("Choose a CSV file with LDR values and tunneling outcomes", type="csv")
        
        if uploaded_file is not None:
            # Read and process uploaded data
            data = pd.read_csv(uploaded_file)
            st.write("### 📊 Data Preview")
            st.write(data.head())
            
            if 'ldr_value' in data.columns and 'tunneled' in data.columns:
                # Prepare and split data
                X = predictor.prepare_data(data['ldr_value'])
                y = data['tunneled']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                # Train model
                predictor.train(X_train, y_train)
                
                # Make predictions on test set
                y_pred, y_prob = predictor.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Display results
                st.write(f"### 🎯 Model Performance")
                st.write(f"Test Accuracy: {accuracy:.2%}")
                
                # Plot results
                fig = px.scatter(
                    data_frame=pd.DataFrame({
                        'LDR Value': X_test['ldr_value'],
                        'Tunneling Probability': y_prob[:, 1],
                        'Actual Outcome': y_test
                    }),
                    x='LDR Value',
                    y='Tunneling Probability',
                    color='Actual Outcome',
                    title='Tunneling Probability vs LDR Value'
                )
                st.plotly_chart(fig)
            else:
                st.error("Please ensure your CSV file has 'ldr_value' and 'tunneled' columns")
            
    elif mode == "Single Prediction":
        st.write("### 🔍 Single Value Prediction")
        ldr_value = st.slider("Enter LDR value", 0.0, 10.0, 5.0)
        
        if st.button("Calculate Probability"):
            probability = predictor.simulate_tunneling(ldr_value)
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                title = {'text': "Tunneling Probability"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ]
                }
            ))
            st.plotly_chart(fig)
            
    else:  # Real-time Simulation
        st.write("### ⚡ Real-time Simulation")
        chart_placeholder = st.empty()
        data = []
        
        for _ in range(50):  # Simulate 50 readings
            ldr_value = random.uniform(0, 10)
            probability = predictor.simulate_tunneling(ldr_value)
            data.append({'LDR Value': ldr_value, 'Probability': probability})
            
            # Update plot
            df = pd.DataFrame(data)
            fig = px.line(df, x=df.index, y='Probability', 
                         title='Real-time Tunneling Probability',
                         labels={'index': 'Time Step', 'Probability': 'Tunneling Probability'})
            chart_placeholder.plotly_chart(fig)
            
            time.sleep(0.1)

if __name__ == "__main__":
    main()
