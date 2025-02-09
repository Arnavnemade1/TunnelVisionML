import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

class QuantumTunnelingPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            random_state=42
        )
        
    def process_data(self, data):
        if isinstance(data, pd.DataFrame):
            values = data['ldr_value'].values.reshape(-1, 1)
        else:
            values = np.array(data).reshape(-1, 1)
        return self.scaler.fit_transform(values)
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

def create_sample_data():
    num_samples = st.sidebar.slider("Number of samples", 10, 100, 50)
    noise_level = st.sidebar.slider("Noise level", 0.0, 1.0, 0.2)
    x = np.linspace(0, 10, num_samples)
    y = np.sin(x) + noise_level * np.random.randn(num_samples)
    return pd.DataFrame({
        'ldr_value': y,
        'tunneling': (y > 0).astype(int)
    })

def main():
    st.title("Quantum Tunneling Predictor")
    st.write("""
    ### LDR-based Quantum Tunneling Probability Prediction
    Use the sidebar to select your data source and control parameters.
    """)
    
    predictor = QuantumTunnelingPredictor()
    
    st.sidebar.title("Controls")
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Sample Data", "Single Value"]
    )
    
    if data_source == "Sample Data":
        df = create_sample_data()
        st.write("Sample Data Preview:")
        st.dataframe(df.head())
        
        processed_data = predictor.process_data(df)
        predictor.model.fit(processed_data, df['tunneling'])
        predictions = predictor.predict(processed_data)
        
        # Using streamlit's native plotting
        df['predictions'] = predictions
        st.line_chart(df[['ldr_value', 'predictions']])
                
    else:  # Single Value
        value = st.number_input("Enter LDR value:", -10.0, 10.0, 0.0)
        if st.button("Predict"):
            # Create sample data for initial training
            train_df = create_sample_data()
            processed_train = predictor.process_data(train_df)
            predictor.model.fit(processed_train, train_df['tunneling'])
            
            # Make prediction
            processed_value = predictor.process_data([[value]])
            prediction = predictor.predict([processed_value[0]])[0]
            
            st.write(f"Tunneling Probability: {prediction:.2%}")
            
            # Visual representation using progress bar
            st.progress(float(prediction))

if __name__ == "__main__":
    main()
