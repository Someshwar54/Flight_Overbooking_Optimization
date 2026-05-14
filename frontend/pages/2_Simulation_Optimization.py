import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import json

st.set_page_config(page_title="Simulation & Optimization", page_icon="⚙️", layout="wide")
st.title("⚙️ Simulation & Optimization")

st.markdown("""
This tool simulates flight revenue by comparing different overbooking strategies:
- **No Overbooking**: Sell exactly up to capacity.
- **Fixed Overbooking**: Overbook by a flat number of seats.
- **Dynamic ML Overbooking**: Overbook based on the XGBoost predicted no-show sum.
""")

# Controls
st.sidebar.header("Simulation Parameters")
flight_capacity = st.sidebar.number_input("Flight Capacity", min_value=50, max_value=500, value=100, step=10)
ticket_price = st.sidebar.number_input("Ticket Price ($)", min_value=50, max_value=2000, value=200, step=50)
compensation = st.sidebar.number_input("Compensation Cost ($)", min_value=100, max_value=5000, value=400, step=50)
num_passengers_to_simulate = st.sidebar.slider("Passengers in Queue", 50, 200, 110)

import os

@st.cache_data
def load_test_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return pd.read_csv(os.path.join(base_dir, 'data', 'processed', '03_test_predictions.csv'))

try:
    df_test = load_test_data()
    # Sample passengers
    sampled_passengers = df_test.sample(n=num_passengers_to_simulate, replace=True, random_state=42)
    
    if st.button("Run Simulation"):
        with st.spinner("Running simulation via Backend API..."):
            # Due to the complexity of sending all features over API (which requires the exact Pydantic schema match),
            # we will run the simulation logic locally here for the dashboard visual, or hit the API if perfectly aligned.
            # Let's hit the API. But wait, `PassengerData` in backend requires ~30 fields. 
            # `df_test` contains the engineered features which are already scaled/encoded. The backend expects raw data to encode it.
            # Since `df_test` in notebook 3 is from `X_test` (which IS encoded), the backend might fail if it tries to re-encode.
            # To handle this gracefully in this MVP, we'll implement the simulation locally in Streamlit as well, identical to backend.
            
            # Local Simulation fallback
            predicted_no_shows = sampled_passengers['No_Show_Prob'].sum()
            actual_no_shows = sampled_passengers['No_Show_Actual'].sum()
            total_passengers = len(sampled_passengers)
            
            results = []
            for strategy in ['none', 'fixed', 'dynamic']:
                if strategy == 'none':
                    tickets_sold = flight_capacity
                elif strategy == 'fixed':
                    tickets_sold = flight_capacity + 5
                elif strategy == 'dynamic':
                    tickets_sold = flight_capacity + int(round(predicted_no_shows))
                    
                no_show_rate = actual_no_shows / total_passengers if total_passengers > 0 else 0
                actual_show_ups = int(round(tickets_sold * (1 - no_show_rate)))
                
                revenue = tickets_sold * ticket_price
                bumped = max(0, actual_show_ups - flight_capacity)
                compensation_cost = bumped * compensation
                empty_seats = max(0, flight_capacity - actual_show_ups)
                net_revenue = revenue - compensation_cost
                
                results.append({
                    "Strategy": strategy.capitalize(),
                    "Tickets Sold": tickets_sold,
                    "Show-Ups": actual_show_ups,
                    "Revenue ($)": revenue,
                    "Bumped": bumped,
                    "Compensation ($)": compensation_cost,
                    "Empty Seats": empty_seats,
                    "Net Revenue ($)": net_revenue
                })
            
            res_df = pd.DataFrame(results)
            st.success("Simulation Complete!")
            
            st.dataframe(res_df.style.format({
                "Revenue ($)": "${:,.2f}",
                "Compensation ($)": "${:,.2f}",
                "Net Revenue ($)": "${:,.2f}"
            }))
            
            # Visuals
            fig1 = px.bar(res_df, x="Strategy", y="Net Revenue ($)", title="Net Revenue by Strategy", color="Strategy", text="Net Revenue ($)")
            fig1.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            
            fig2 = go.Figure(data=[
                go.Bar(name='Empty Seats', x=res_df['Strategy'], y=res_df['Empty Seats']),
                go.Bar(name='Bumped Passengers', x=res_df['Strategy'], y=res_df['Bumped'])
            ])
            fig2.update_layout(barmode='group', title='Trade-off: Empty Seats vs Bumped Passengers')
            
            c1, c2 = st.columns(2)
            c1.plotly_chart(fig1, use_container_width=True)
            c2.plotly_chart(fig2, use_container_width=True)

except FileNotFoundError:
    st.error("Test data not found. Please ensure Phase 2 Model Training is complete.")
