import streamlit as st
import math
import random

def simulate_tunneling(ldr_value):
    # Simple simulation based on a sine wave and the input value
    base_probability = (math.sin(ldr_value) + 1) / 2  # Returns value between 0 and 1
    # Add some random noise
    noise = random.uniform(-0.1, 0.1)
    probability = max(0, min(1, base_probability + noise))  # Ensure between 0 and 1
    return probability

def main():
    st.title("Quantum Tunneling Simulator")
    st.write("""
    ### Simple LDR-based Quantum Tunneling Demonstration
    This is a basic simulation of quantum tunneling probability based on LDR values.
    """)
    
    # Sidebar for mode selection
    mode = st.sidebar.radio("Select Mode", ["Single Value", "Real-time Simulation"])
    
    if mode == "Single Value":
        # Single value prediction
        ldr_value = st.slider("Enter LDR value", 0.0, 10.0, 5.0)
        if st.button("Calculate Probability"):
            probability = simulate_tunneling(ldr_value)
            
            # Display result
            st.write(f"Tunneling Probability: {probability:.2%}")
            st.progress(probability)
            
            # Add some context
            if probability > 0.7:
                st.success("High probability of tunneling!")
            elif probability > 0.3:
                st.info("Moderate probability of tunneling")
            else:
                st.warning("Low probability of tunneling")
    
    else:
        # Real-time simulation
        st.write("Simulating real-time LDR readings...")
        import time
        
        # Create placeholder for updating values
        value_placeholder = st.empty()
        prob_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        # Simulate 5 readings
        for i in range(5):
            ldr_value = random.uniform(0, 10)
            probability = simulate_tunneling(ldr_value)
            
            value_placeholder.write(f"LDR Reading: {ldr_value:.2f}")
            prob_placeholder.write(f"Tunneling Probability: {probability:.2%}")
            progress_placeholder.progress(probability)
            
            time.sleep(1)  # Wait 1 second between readings
        
        st.write("Simulation complete!")

if __name__ == "__main__":
    main()

