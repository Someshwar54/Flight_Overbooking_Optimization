import streamlit as st
import requests

st.set_page_config(
    page_title="Flight Overbooking Optimizer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✈️ Flight Overbooking Optimization System")

st.markdown("""
Welcome to the AI-Powered **Flight No-Show Prediction and Dynamic Overbooking Optimization System**.

### 🌟 Key Features
- **No-Show Prediction**: Predict the probability of passengers missing their flight using XGBoost.
- **Dynamic Overbooking**: ML-driven recommendations for optimal overbooking strategy.
- **Revenue Simulation**: Compare strategies to maximize net profit while minimizing denied boardings.

Use the sidebar to navigate between different views:
1. **Overview Dashboard**: High-level metrics.
2. **Flight No-Show Risk**: Deep dive into passenger risk.
3. **Simulation & Optimization**: Compare strategies and calculate revenue.
""")

st.sidebar.success("Select a page above.")

try:
    res = requests.get("http://127.0.0.1:8000/")
    if res.status_code == 200:
        st.sidebar.success("Backend API Connected")
    else:
        st.sidebar.error("Backend API Error")
except:
    st.sidebar.warning("Backend API Not Reachable. Make sure FastAPI is running.")
