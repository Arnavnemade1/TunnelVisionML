import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math
import random

class QuantumTunnelingPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def parse_experimental_data(self, text_data):
        lines = text_data.split('\n')
        data = []
        barrier_data = []

        for i, line in enumerate(lines):
            try:
                if 'LDR Value:' in line and '|' in line:
                    parts = line.split('|')
                    ldr_value = float(parts[0].split(':')[1].strip())
                    threshold = float(parts[1].split(':')[1].strip())
                    random_val = float(parts[2].split(':')[1].strip())
                    probability = float(parts[3].split(':')[1].strip())
                    tunneling = 'YES' if 'YES' in parts[4] else 'NO'

                    data.append({
                        'measurement_id': i,
                        'ldr_value': ldr_value,
                        'threshold': threshold,
                        'random': random_val,
                        'probability': probability,
                        'tunneling': tunneling,
                        'timestamp': pd.Timestamp.now() - pd.Timedelta(seconds=(len(data)))
                    })
                elif 'Barrier:' in line:
                    barrier_count = line.count('#')
                    dot_count = line.count('.')
                    barrier_data.append({
                        'measurement_id': len(barrier_data),
                        'barrier_strength': barrier_count,
                        'barrier_gaps': dot_count
                    })
            except Exception as e:
                print(f"Error parsing line {i}: {e}")

        return pd.DataFrame(data), pd.DataFrame(barrier_data)

    def simulate_tunneling(self, ldr_value):
        base_probability = (math.sin(ldr_value / 100) + 1) / 2
        noise = random.uniform(-0.1, 0.1)
        potential_barrier = math.exp(-ldr_value / 500)
        quantum_factor = 1 / (1 + math.exp(-(ldr_value - 500) / 100))
        probability = max(0, min(1, (base_probability + noise) * potential_barrier * quantum_factor))
        return probability

def create_heatmap(df):
    if df.empty or 'ldr_value' not in df.columns or 'random' not in df.columns or 'probability' not in df.columns:
        return go.Figure()  # Return an empty figure if data is missing

    heatmap_data = df.pivot_table(
        values='probability',
        index=pd.qcut(df['ldr_value'], 10, duplicates='drop'),
        columns=pd.qcut(df['random'], 10, duplicates='drop'),
        aggfunc='mean'
    )

    return go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=np.arange(len(heatmap_data.columns)),
        y=np.arange(len(heatmap_data.index)),
        colorscale='Viridis',
        colorbar=dict(title='Probability')
    ))

def main():
    st.title("Quantum Tunneling Analyzer")
    predictor = QuantumTunnelingPredictor()

    uploaded_file = st.file_uploader("Upload data (TXT or CSV)", type=['txt', 'csv'])
    if uploaded_file:
        raw_data = uploaded_file.read().decode()
        df, barrier_df = predictor.parse_experimental_data(raw_data)

        if not df.empty:
            st.write("## Data Preview")
            st.dataframe(df.head())

            st.write("## Heatmap")
            heatmap_fig = create_heatmap(df)
            st.plotly_chart(heatmap_fig)

if __name__ == "__main__":
    main()
