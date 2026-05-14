# %% [markdown]
# # 03: Model Training
# In this notebook, we train machine learning models to predict the `No_Show` probability.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

# %% [markdown]
# ## 1. Prepare Data

# %%
df = pd.read_csv('../data/processed/02_engineered_data.csv')

X = df.drop(columns=['No_Show'])
y = df['No_Show']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for backend
joblib.dump(scaler, '../artifacts/scaler.pkl')

# %% [markdown]
# ## 2. Train Models

# %%
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []

for name, model in models.items():
    print(f"Training {name}...")
    # Train
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    roc_auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "ROC-AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

results_df = pd.DataFrame(results)
print("\n--- Model Comparison ---")
print(results_df)

# %% [markdown]
# ## 3. Select and Save Best Model
# Based on tabular data performance, XGBoost or Random Forest usually perform best. 
# We'll select XGBoost.

# %%
best_model = models["XGBoost"]
joblib.dump(best_model, '../artifacts/xgboost_no_show_model.pkl')
print("\nSaved best model (XGBoost) to /artifacts/xgboost_no_show_model.pkl")

# Save test set for later simulation
test_data = X_test.copy()
test_data['No_Show_Actual'] = y_test
test_data['No_Show_Prob'] = best_model.predict_proba(X_test)[:, 1]
test_data.to_csv('../data/processed/03_test_predictions.csv', index=False)
