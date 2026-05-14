# %% [markdown]
# # 01: Data Cleaning and Merging
# In this notebook, we load the three datasets, clean them, and merge them into a single analytical dataset.

# %%
import pandas as pd
import numpy as np

# Load Datasets
df_synthetic = pd.read_csv('../Datasets/synthetic_flight_passenger_data.csv')
df_booking = pd.read_csv('../Datasets/customer_booking.csv', encoding='latin1')
df_ontime = pd.read_csv('../Datasets/T_ONTIME_MARKETING.csv')

# %% [markdown]
# ## 1. Clean Synthetic Passenger Data
# This is our primary dataset because it contains the target variable `No_Show`.

# %%
df_synthetic.info()

# Create a common 'Route' column to merge with booking data
df_synthetic['Route'] = df_synthetic['Departure_Airport'] + df_synthetic['Arrival_Airport']
# Handle Missing Values
df_synthetic.fillna({
    'Delay_Minutes': 0,
    'Weather_Impact': 0,
}, inplace=True)

# %% [markdown]
# ## 2. Clean Customer Booking Data
# We aggregate the booking data at the route level to extract features that describe route popularity and typical booking behavior.

# %%
df_booking.info()

# Aggregate by route
route_features = df_booking.groupby('route').agg({
    'num_passengers': 'mean',
    'purchase_lead': 'mean',
    'length_of_stay': 'mean',
    'wants_extra_baggage': 'mean',
    'wants_preferred_seat': 'mean',
    'wants_in_flight_meals': 'mean',
    'booking_complete': 'mean' # Conversion rate for the route
}).reset_index()

route_features.rename(columns={
    'route': 'Route',
    'num_passengers': 'route_avg_passengers',
    'purchase_lead': 'route_avg_purchase_lead',
    'length_of_stay': 'route_avg_length_of_stay',
    'wants_extra_baggage': 'route_prop_extra_baggage',
    'wants_preferred_seat': 'route_prop_preferred_seat',
    'wants_in_flight_meals': 'route_prop_in_flight_meals',
    'booking_complete': 'route_booking_conversion_rate'
}, inplace=True)

# %% [markdown]
# ## 3. Merge Datasets

# %%
# Merge synthetic with aggregated booking data
df_merged = pd.merge(df_synthetic, route_features, on='Route', how='left')

# Since some routes in synthetic data may not exist in booking data, we'll fill NaN with median/mean values
fill_values = {col: route_features[col].median() for col in route_features.columns if col != 'Route'}
df_merged.fillna(fill_values, inplace=True)

# %% [markdown]
# Note: T_ONTIME_MARKETING provides ID mappings but lacks airport string codes in the immediate view. 
# We'll skip a direct join with T_ONTIME_MARKETING to avoid dropping rows since the synthetic data is already rich enough.
# Instead, we rely on the route-level aggregates from `customer_booking.csv`.

# %%
print(f"Final merged dataset shape: {df_merged.shape}")
df_merged.to_csv('../data/processed/01_merged_data.csv', index=False)
print("Saved merged dataset to /data/processed/01_merged_data.csv")
