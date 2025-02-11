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
    
    def parse_experimental_data(self, text_data):
        """Parse the specific format of quantum tunneling data"""
        lines = text_data.split('\n')
        data = []
        barrier_data = []
        
        for i, line in enumerate(lines):
            if 'LDR Value:' in line and '|' in line:
                try:
                    parts = line.split('|')
                    ldr_value = float(parts[0].split(':')[1].strip())
                    threshold = float(parts[1].split(':')[1].strip())
                    random_val = float(parts[2].split(':')[1].strip())
                    probability = float(parts[3].split(':')[1].strip())
                    tunneling = 'YES' in parts[4] if len(parts) > 4 else 'NO'
                    
                    data.append({
                        'measurement_id': i,
                        'ldr_value': ldr_value,
                        'threshold': threshold,
                        'random': random_val,
                        'probability': probability,
                        'tunneling': tunneling
                    })
                except:
                    continue
            
            elif 'Barrier:' in line:
                barrier = line.split(':')[1].strip()
                barrier_count = barrier.count('#')
                dot_count = barrier.count('.')
                barrier_data.append({
                    'measurement_id': len(barrier_data),
                    'barrier_strength': barrier_count,
                    'barrier_gaps': dot_count
                })
                
        return pd.DataFrame(data), pd.DataFrame(barrier_data)
    
    def simulate_tunneling(self, ldr_value):
        """Simulate tunneling probability for a given LDR value"""
        base_probability = (math.sin(ldr_value/100) + 1) / 2
        noise = random.uniform(-0.1, 0.1)
        potential_barrier = math.exp(-ldr_value / 500)
        probability = max(0, min(1, (base_probability + noise) * potential_barrier))
        return probability
    
    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True

def main():
    st.title("🌌 Quantum Tunneling Analyzer")
    
    # Sidebar for mode selection
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Select Mode", ["Upload Data", "Single Prediction", "Real-time Simulation"])
    
    predictor = QuantumTunnelingPredictor()
    
    if mode == "Upload Data":
        st.write("""
        ### 📤 Upload Experimental Data
        Upload your quantum tunneling experiment data to visualize and analyze the results.
        """)
        
        uploaded_file = st.file_uploader("Upload your experiment data (TXT)", type=['txt'])
        
        if uploaded_file is not None:
            raw_data = uploaded_file.read().decode()
            df, barrier_df = predictor.parse_experimental_data(raw_data)
            
            if not df.empty:
                st.success(f"Successfully loaded {len(df)} measurements!")
                
                # Data Overview
                st.header("📊 Data Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Measurements", len(df))
                with col2:
                    tunneling_count = df['tunneling'].value_counts().get('YES', 0)
                    tunneling_rate = (tunneling_count / len(df)) * 100
                    st.metric("Tunneling Events", f"{tunneling_count} ({tunneling_rate:.1f}%)")
                with col3:
                    avg_ldr = df['ldr_value'].mean()
                    st.metric("Average LDR", f"{avg_ldr:.1f}")

                # Main Visualization
                st.header("📈 Quantum Tunneling Visualization")
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['measurement_id'],
                    y=df['ldr_value'],
                    name='LDR Value',
                    mode='lines+markers',
                    line=dict(color='blue'),
                    marker=dict(
                        size=8,
                        color=df['tunneling'].map({'YES': 'red', 'NO': 'blue'})
                    )
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['measurement_id'],
                    y=df['threshold'],
                    name='Threshold',
                    mode='lines',
                    line=dict(color='red', dash='dash')
                ))
                
                if not barrier_df.empty:
                    fig.add_trace(go.Bar(
                        x=barrier_df['measurement_id'],
                        y=barrier_df['barrier_strength'],
                        name='Barrier Strength',
                        opacity=0.3,
                        marker_color='gray'
                    ))
                
                fig.update_layout(
                    title="Quantum Tunneling Experiment Results",
                    xaxis_title="Measurement Index",
                    yaxis_title="Value",
                    height=600,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig)

                # Probability Analysis
                st.header("🎲 Tunneling Probability Analysis")
                fig_prob = px.scatter(
                    df,
                    x='random',
                    y='probability',
                    color='tunneling',
                    title='Tunneling Probability vs Random Value'
                )
                st.plotly_chart(fig_prob)

                # Data Table
                st.header("📋 Raw Data")
                st.dataframe(df.style.highlight_max(axis=0))

                # Export functionality
                st.download_button(
                    label="📥 Download CSV",
                    data=df.to_csv(index=False),
                    file_name="tunneling_analysis.csv",
                    mime="text/csv"
                )

    elif mode == "Single Prediction":
        st.write("### 🔍 Single Value Prediction")
        ldr_value = st.slider("Enter LDR value", 0, 1023, 450)
        
        if st.button("Calculate Probability"):
            probability = predictor.simulate_tunneling(ldr_value)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Tunneling Probability"},
                gauge={
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
            
            # Barrier Visualization
            barrier_length = 50
            barrier_strength = int((1 - probability) * barrier_length)
            barrier = '#' * barrier_strength + '.' * (barrier_length - barrier_strength)
            st.code(f"Barrier: {barrier}", language=None)

    else:  # Real-time Simulation
        st.write("### ⚡ Real-time Simulation")
        chart_placeholder = st.empty()
        data = []
        
        if st.button("Start Simulation"):
            for _ in range(50):
                ldr_value = random.uniform(0, 1023)
                probability = predictor.simulate_tunneling(ldr_value)
                data.append({
                    'LDR Value': ldr_value,
                    'Probability': probability,
                    'Timestamp': len(data)
                })
                
                df = pd.DataFrame(data)
                fig = px.line(
                    df, 
                    x='Timestamp',
                    y=['LDR Value', 'Probability'],
                    title='Real-time Tunneling Simulation'
                )
                chart_placeholder.plotly_chart(fig)
                
                time.sleep(0.1)

if __name__ == "__main__":
    main()
