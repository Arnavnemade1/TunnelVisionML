import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.figure_factory as ff
import math
import random
import time
from io import StringIO
from scipy import stats
import streamlit.components.v1 as components  # For custom HTML/CSS

# Set page config
st.set_page_config(page_title="Quantum Tunneling Analyzer", page_icon="🌌", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4CAF50;
    }
    .stMarkdown h1 {
        font-size: 36px;
    }
    .stMarkdown h2 {
        font-size: 28px;
    }
    .stMarkdown h3 {
        font-size: 22px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
                except Exception as e:
                    st.warning(f"Skipping malformed line {i}: {line}. Error: {str(e)}")
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
        """Simulate tunneling probability with enhanced physics modeling"""
        base_probability = (math.sin(ldr_value / 100) + 1) / 2
        noise = random.uniform(-0.1, 0.1)
        potential_barrier = math.exp(-ldr_value / 500)
        quantum_factor = 1 / (1 + math.exp(-(ldr_value - 500) / 100))
        probability = max(0, min(1, (base_probability + noise) * potential_barrier * quantum_factor))
        return probability

def create_heatmap(df):
    """Create a 2D heatmap of tunneling events"""
    try:
        heatmap_data = df.pivot_table(
            values='probability',
            index=pd.qcut(df['ldr_value'], 10),
            columns=pd.qcut(df['random'], 10),
            aggfunc='mean'
        )
        return go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=np.arange(10),
            y=np.arange(10),
            colorscale='Viridis',
            colorbar=dict(title='Probability')
        ))
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        return None

def create_distribution_plot(tunneling_data, no_tunneling_data):
    """Create a distribution plot for tunneling probabilities"""
    if len(tunneling_data) > 0 and len(no_tunneling_data) > 0:
        try:
            return ff.create_distplot(
                [tunneling_data, no_tunneling_data],
                ['Tunneling', 'No Tunneling'],
                bin_size=0.02
            )
        except Exception as e:
            st.error(f"Error creating distribution plot: {str(e)}")
            return None
    else:
        st.warning("Insufficient data for distribution plot. Ensure both tunneling and no tunneling data are available.")
        return None

def perform_statistical_tests(tunneling_data, no_tunneling_data):
    """Perform statistical tests on the data"""
    results = {}
    
    st.write(f"Tunneling data points: {len(tunneling_data)}")
    st.write(f"No tunneling data points: {len(no_tunneling_data)}")
    
    if len(tunneling_data) > 0 and len(no_tunneling_data) > 0:
        try:
            ks_stat, ks_pval = stats.ks_2samp(tunneling_data, no_tunneling_data)
            results['ks_test'] = {
                'statistic': ks_stat,
                'p_value': ks_pval
            }
        except Exception as e:
            results['ks_test'] = None
            st.error(f"Error performing KS test: {str(e)}")
    else:
        results['ks_test'] = None
        st.error("Insufficient data for statistical tests. Ensure both tunneling and no tunneling data are available.")
    
    return results

def generate_summary(stats_results):
    """Generate a human-readable summary of the statistical test results"""
    summary = ""
    
    if stats_results.get('ks_test'):
        ks_stat = stats_results['ks_test']['statistic']
        ks_pval = stats_results['ks_test']['p_value']
        
        if ks_pval < 0.05:
            summary += (
                "📊 **Significant Difference Detected**:\n"
                "The Kolmogorov-Smirnov test indicates a **statistically significant difference** "
                f"(p-value = {ks_pval:.4f}) between the tunneling and no tunneling probability distributions. "
                "This suggests that the two groups are likely drawn from different underlying distributions."
            )
        else:
            summary += (
                "📊 **No Significant Difference Detected**:\n"
                "The Kolmogorov-Smirnov test does **not** indicate a statistically significant difference "
                f"(p-value = {ks_pval:.4f}) between the tunneling and no tunneling probability distributions. "
                "This suggests that the two groups may be drawn from similar underlying distributions."
            )
    else:
        summary += "📊 **No Statistical Results Available**:\nUnable to generate a summary due to insufficient data."
    
    return summary

def main():
    st.title("🌌 Enhanced Quantum Tunneling Analyzer")
    
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Select Mode", ["Upload Data", "Single Prediction", "Real-time Simulation", "Advanced Analytics"])
    
    predictor = QuantumTunnelingPredictor()
    
    if mode == "Upload Data":
        st.write("""
        ### 📤 Upload Experimental Data
        Upload your quantum tunneling experiment data for advanced visualization and analysis.
        """)
        
        uploaded_file = st.file_uploader("Upload your experiment data (TXT or CSV)", type=['txt', 'csv'])
        
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                raw_data = uploaded_file.read().decode()
                df, barrier_df = predictor.parse_experimental_data(raw_data)
            elif uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                barrier_df = pd.DataFrame()  # Placeholder for CSV data
            
            if not df.empty:
                st.header("📊 Data Overview")
                st.write(f"Total measurements: {len(df)}")
                st.write(f"Tunneling events: {df['tunneling'].value_counts().get('YES', 0)}")
                
                # Main Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['measurement_id'],
                    y=df['ldr_value'],
                    name='LDR Value',
                    mode='lines+markers'
                ))
                st.plotly_chart(fig, use_container_width=True)

    elif mode == "Single Prediction":
        st.write("### 🔍 Single Value Prediction")
        
        # Add a rotary wheel for LDR value selection
        ldr_value = st.slider("Enter LDR value", 0, 1023, 450, key="ldr_slider")
        
        # Display the rotary wheel using HTML/CSS
        components.html(
            f"""
            <style>
            .wheel {{
                width: 200px;
                height: 200px;
                border-radius: 50%;
                background: conic-gradient(
                    from 0deg,
                    #ff7f50 0% {ldr_value / 1023 * 100}%,
                    #ddd {ldr_value / 1023 * 100}% 100%
                );
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
            </style>
            <div class="wheel">{ldr_value}</div>
            """,
            height=220,
        )
        
        if st.button("Calculate"):
            probability = predictor.simulate_tunneling(ldr_value)
            st.success(f"Tunneling Probability: **{probability:.4f}**")

    elif mode == "Real-time Simulation":
        st.write("### ⚡ Real-time Simulation")
        chart_placeholder = st.empty()
        data = []
        
        if st.button("Start Simulation"):
            for _ in range(50):
                ldr_value = random.uniform(0, 1023)
                probability = predictor.simulate_tunneling(ldr_value)
                data.append({
                    'time': len(data),
                    'ldr_value': ldr_value,
                    'probability': probability
                })
                
                df = pd.DataFrame(data)
                fig = px.line(df, x='time', y=['ldr_value', 'probability'])
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(0.1)

    else:  # Advanced Analytics
        st.write("### 🔬 Advanced Analytics")
        
        num_samples = st.slider("Number of Samples", 100, 1000, 500)
        
        if st.button("Run Analysis"):
            # Generate synthetic data
            ldr_values = np.random.uniform(0, 1023, num_samples)
            probabilities = np.array([predictor.simulate_tunneling(ldr) for ldr in ldr_values])
            tunneling = probabilities > np.random.random(num_samples)
            
            # Create DataFrame
            df = pd.DataFrame({
                'ldr_value': ldr_values,
                'probability': probabilities,
                'tunneling': tunneling
            })
            
            # Split data
            tunneling_probs = df[df['tunneling']]['probability'].values
            no_tunneling_probs = df[~df['tunneling']]['probability'].values
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution plot
                fig = create_distribution_plot(tunneling_probs, no_tunneling_probs)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Scatter plot
                fig = px.scatter(df, x='ldr_value', y='probability', color=df['tunneling'].astype(str))
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical tests
            st.subheader("📊 Statistical Analysis")
            stats_results = perform_statistical_tests(tunneling_probs, no_tunneling_probs)
            
            if stats_results.get('ks_test'):
                st.markdown("""
                ### Kolmogorov-Smirnov Test Results
                """)
                st.markdown(f"""
                - **Statistic**: `{stats_results['ks_test']['statistic']:.4f}`
                - **P-value**: `{stats_results['ks_test']['p_value']:.4f}`
                """)
                
                # Generate and display the summary
                summary = generate_summary(stats_results)
                st.markdown("### 📝 Summary of Findings")
                st.markdown(summary)

if __name__ == "__main__":
    main()
