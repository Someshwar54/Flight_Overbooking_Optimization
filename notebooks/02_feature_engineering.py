# %% [markdown]
# # 02: Feature Engineering
# In this notebook, we engineer features for our `No_Show` prediction model.

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load Merged Dataset
df = pd.read_csv('../data/processed/01_merged_data.csv')

# %% [markdown]
# ## 1. Temporal Features

# %%
# Convert Departure_Time to datetime
df['Departure_Time'] = pd.to_datetime(df['Departure_Time'])

df['Departure_Month'] = df['Departure_Time'].dt.month
df['Departure_DayOfWeek'] = df['Departure_Time'].dt.dayofweek
df['Departure_Hour'] = df['Departure_Time'].dt.hour
df['Is_Weekend'] = df['Departure_DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# %% [markdown]
# ## 2. Route & Airline Features
# Compute historical risk (No-Show rate) for airlines and routes. 
# Note: In a strict ML pipeline, this should be done inside cross-validation, but for simplicity we calculate it globally here or use standard categorical encoding.
# To avoid target leakage, we'll use frequency encoding or target encoding with smoothing. For simplicity, we'll use Label Encoding for tree-based models and later target encoding in the pipeline if needed.

# %%
# Frequency Encoding for Route
route_freq = df['Route'].value_counts() / len(df)
df['Route_Freq'] = df['Route'].map(route_freq)

# Categorical columns to encode
cat_cols = ['Airline', 'Departure_Airport', 'Arrival_Airport', 'Route', 
            'Gender', 'Income_Level', 'Travel_Purpose', 'Seat_Class', 
            'Frequent_Flyer_Status', 'Check_in_Method', 'Seat_Selected', 'Flight_Status']

# Handle NaN in Frequent_Flyer_Status
df['Frequent_Flyer_Status'] = df['Frequent_Flyer_Status'].fillna('None')

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# %% [markdown]
# ## 3. Final Feature Selection

# %%
# Drop identifiers and datetime
drop_cols = ['Passenger_ID', 'Flight_ID', 'Departure_Time']
df_features = df.drop(columns=drop_cols)

print(df_features.info())

# Save engineered data
df_features.to_csv('../data/processed/02_engineered_data.csv', index=False)
print("Saved engineered dataset to /data/processed/02_engineered_data.csv")

# We will also save the label encoders for the backend API later.
import joblib
joblib.dump(encoders, '../artifacts/label_encoders.pkl')
print("Saved Label Encoders to /artifacts/label_encoders.pkl")
