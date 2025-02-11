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
                # Parse measurement data
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
            
            # Parse barrier visualization
            elif 'Barrier:' in line:
                barrier = line.split(':')[1].strip()
                barrier_count = barrier.count('#')
                dot_count = barrier.count('.')
                barrier_data.append({
                    'measurement_id': len(barrier_data),
                    'barrier_strength': barrier_count,
                    'barrier_gaps': dot_count
                })
                
        df = pd.DataFrame(data)
        barrier_df = pd.DataFrame(barrier_data)
        
        return df, barrier_df

def main():
    st.title("🌌 Quantum Tunneling Analyzer")
    st.write("""
    ### Experimental Data Visualization
    Upload your quantum tunneling experiment data to visualize the results.
    """)
    
    predictor = QuantumTunnelingPredictor()
    
    # File upload
    uploaded_file = st.file_uploader("Upload your experiment data (TXT)", type=['txt'])
    
    if uploaded_file is not None:
        # Read and process data
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

            # Main Experiment Visualization
            st.header("📈 Quantum Tunneling Visualization")
            
            # Create combined visualization
            fig = go.Figure()
            
            # Add LDR values
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
            
            # Add threshold line
            fig.add_trace(go.Scatter(
                x=df['measurement_id'],
                y=df['threshold'],
                name='Threshold',
                mode='lines',
                line=dict(color='red', dash='dash')
            ))
            
            # Add barrier strength visualization if available
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
                title='Tunneling Probability vs Random Value',
                labels={
                    'random': 'Random Value',
                    'probability': 'Tunneling Probability',
                    'tunneling': 'Tunneling Occurred'
                }
            )
            st.plotly_chart(fig_prob)

            # Data Table
            st.header("📋 Raw Data")
            st.dataframe(df.style.highlight_max(axis=0))

            # Export functionality
            st.header("💾 Export Data")
            if st.button("Export to CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="tunneling_analysis.csv",
                    mime="text/csv"
                )

        else:
            st.error("Could not parse the data. Please check the file format.")

if __name__ == "__main__":
    main()
