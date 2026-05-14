# %% [markdown]
# # 04: Revenue Simulation
# In this notebook, we simulate the financial impact of three strategies:
# 1. No Overbooking
# 2. Fixed Overbooking (e.g., +5 seats per flight)
# 3. Dynamic ML Overbooking (based on No_Show probabilities)

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# ## 1. Load Test Predictions

# %%
df = pd.read_csv('../data/processed/03_test_predictions.csv')

# For simulation, let's group passengers into "Flights"
# Since our data doesn't have an explicit flight identifier after dropping it, 
# we'll simulate flights by grouping every 100 passengers into a "Flight"
df['Simulated_Flight_ID'] = np.arange(len(df)) // 100

# %% [markdown]
# ## 2. Simulation Logic

# %%
FLIGHT_CAPACITY = 100
TICKET_PRICE = 200      # Revenue per ticket
COMPENSATION = 400      # Cost per bumped passenger

def simulate_flight(group, strategy='none', fixed_overbook=0):
    """
    Simulates a flight given passenger probabilities.
    Returns: Revenue, Bumped Passengers, Empty Seats
    """
    total_passengers = len(group)
    actual_no_shows = group['No_Show_Actual'].sum()
    predicted_no_shows = group['No_Show_Prob'].sum()
    
    if strategy == 'none':
        tickets_sold = FLIGHT_CAPACITY
    elif strategy == 'fixed':
        tickets_sold = FLIGHT_CAPACITY + fixed_overbook
    elif strategy == 'dynamic':
        tickets_sold = FLIGHT_CAPACITY + int(round(predicted_no_shows))
    else:
        tickets_sold = FLIGHT_CAPACITY
        
    # Assume actual passengers who showed up out of the tickets_sold
    # For simplicity, we assume the no-show rate of the extra sold tickets is the same as the group average
    no_show_rate = actual_no_shows / total_passengers if total_passengers > 0 else 0
    actual_show_ups = int(round(tickets_sold * (1 - no_show_rate)))
    
    # Calculate costs
    revenue = tickets_sold * TICKET_PRICE
    
    bumped = max(0, actual_show_ups - FLIGHT_CAPACITY)
    compensation_cost = bumped * COMPENSATION
    
    empty_seats = max(0, FLIGHT_CAPACITY - actual_show_ups)
    
    net_revenue = revenue - compensation_cost
    
    return pd.Series({
        'Tickets_Sold': tickets_sold,
        'Actual_Show_Ups': actual_show_ups,
        'Revenue': revenue,
        'Bumped': bumped,
        'Compensation_Cost': compensation_cost,
        'Empty_Seats': empty_seats,
        'Net_Revenue': net_revenue
    })

# %% [markdown]
# ## 3. Compare Strategies

# %%
results_none = df.groupby('Simulated_Flight_ID').apply(lambda g: simulate_flight(g, 'none'))
results_fixed = df.groupby('Simulated_Flight_ID').apply(lambda g: simulate_flight(g, 'fixed', 5))
results_dynamic = df.groupby('Simulated_Flight_ID').apply(lambda g: simulate_flight(g, 'dynamic'))

summary = pd.DataFrame({
    'Strategy': ['No Overbooking', 'Fixed Overbooking (+5)', 'Dynamic ML Overbooking'],
    'Total_Net_Revenue': [results_none['Net_Revenue'].sum(), results_fixed['Net_Revenue'].sum(), results_dynamic['Net_Revenue'].sum()],
    'Total_Bumped': [results_none['Bumped'].sum(), results_fixed['Bumped'].sum(), results_dynamic['Bumped'].sum()],
    'Total_Empty_Seats': [results_none['Empty_Seats'].sum(), results_fixed['Empty_Seats'].sum(), results_dynamic['Empty_Seats'].sum()]
})

print(summary)
summary.to_csv('../data/processed/04_simulation_summary.csv', index=False)
