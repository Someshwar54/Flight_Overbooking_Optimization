import joblib
import pandas as pd
import numpy as np
from ..schemas import PassengerData

import os

# Construct absolute paths relative to this file's directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
artifacts_dir = os.path.join(base_dir, 'artifacts')

try:
    model = joblib.load(os.path.join(artifacts_dir, 'xgboost_no_show_model.pkl'))
    scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.pkl'))
    encoders = joblib.load(os.path.join(artifacts_dir, 'label_encoders.pkl'))
except FileNotFoundError:
    model, scaler, encoders = None, None, None
    print("Warning: ML artifacts not found. Please run notebooks 1-3.")

def predict_no_show_prob(passenger: PassengerData) -> float:
    if model is None:
        return 0.0

    # Convert to DataFrame
    df = pd.DataFrame([passenger.dict()])
    
    # Apply encoders
    for col, le in encoders.items():
        if col in df.columns:
            # Handle unknown labels by assigning a default or most common if needed
            # For simplicity, if it's unknown, we map to the first class
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
            
    # XGBoost doesn't strictly need scaled data, but if we used standard scaler for other models we should match training
    # Actually, we trained XGBoost on UN-SCALED data in notebook 3 (X_train, not X_train_scaled).
    # Let's verify: `model.fit(X_train, y_train)` was used for XGBoost. 
    # So we DO NOT scale for XGBoost.
    
    # Make sure columns match training
    expected_cols = model.feature_names_in_
    df = df[expected_cols]
    
    prob = model.predict_proba(df)[0, 1]
    return float(prob)

def run_simulation(flight_capacity: int, ticket_price: float, compensation: float, passengers: list[PassengerData]):
    if not passengers:
        return []
        
    probs = [predict_no_show_prob(p) for p in passengers]
    predicted_no_shows = sum(probs)
    
    total_passengers = len(passengers)
    # We don't have actuals, so we simulate based on probabilities for the demo
    # We will sample actual no-shows using the probabilities
    actual_no_shows = sum(np.random.rand() < p for p in probs)
    
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
            "strategy": strategy,
            "tickets_sold": tickets_sold,
            "actual_show_ups": actual_show_ups,
            "revenue": revenue,
            "bumped": bumped,
            "compensation_cost": compensation_cost,
            "empty_seats": empty_seats,
            "net_revenue": net_revenue
        })
        
    return results
