import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Overview Dashboard", page_icon="📊", layout="wide")

st.title("📊 Overview Dashboard")

import os

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return pd.read_csv(os.path.join(base_dir, 'data', 'processed', '01_merged_data.csv'))

try:
    df = load_data()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flights (Simulated)", len(df['Flight_ID'].unique()))
    col2.metric("Total Passengers", f"{len(df):,}")
    col3.metric("Overall No-Show Rate", f"{(df['No_Show'].mean() * 100):.2f}%")
    col4.metric("Avg Ticket Price", f"${df['Price_USD'].mean():.2f}")
    
    st.markdown("---")
    
    # Visualizations
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("No-Show Rate by Airline")
        airline_ns = df.groupby('Airline')['No_Show'].mean().reset_index()
        fig_airline = px.bar(airline_ns, x='Airline', y='No_Show', title="Average No-Show Rate by Airline",
                             labels={"No_Show": "No-Show Rate"}, color='No_Show', color_continuous_scale='Reds')
        fig_airline.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig_airline, use_container_width=True)
        
    with col_b:
        st.subheader("No-Show Rate by Booking Lead Time")
        # Bin lead time
        df['Lead_Time_Bin'] = pd.cut(df['Booking_Days_In_Advance'], bins=[0, 7, 14, 30, 60, 365], labels=['0-7', '8-14', '15-30', '31-60', '60+'])
        lead_time_ns = df.groupby('Lead_Time_Bin')['No_Show'].mean().reset_index()
        fig_lead = px.line(lead_time_ns, x='Lead_Time_Bin', y='No_Show', title="No-Show Rate vs. Days in Advance", markers=True)
        fig_lead.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig_lead, use_container_width=True)

except FileNotFoundError:
    st.error("Data not found. Please ensure Phase 1 data processing is complete.")
