# ✈️ Flight No-Show Prediction & Dynamic Overbooking Optimization

An end-to-end AI-powered revenue optimization platform for airlines. This system predicts passenger no-show probabilities using machine learning and recommends optimal overbooking levels to maximize revenue while minimizing denied boarding risks.

## 🚀 Overview
Airlines often face revenue loss due to empty seats from passenger no-shows. This project provides a production-grade solution that:
1. **Predicts** no-show risks at the individual passenger level.
2. **Optimizes** overbooking counts using a dynamic ML-driven engine.
3. **Simulates** financial impact comparing different business strategies.

## 🛠️ Tech Stack
- **Backend**: FastAPI (Python), SQLAlchemy, SQLite
- **Frontend**: Streamlit, Plotly
- **Machine Learning**: XGBoost, Scikit-Learn, Pandas, NumPy
- **Infrastructure**: Docker, Docker Compose
- **Analysis**: Jupyter Notebooks

## 📂 Project Structure
```bash
├── artifacts/             # Trained models and encoders
├── backend/               # FastAPI application
│   ├── models/            # SQLAlchemy database models
│   ├── routes/            # API endpoints
│   ├── services/          # Business logic (ML & Simulation)
│   └── main.py            # Entry point
├── data/                  # Raw and processed datasets
├── frontend/              # Streamlit dashboard
│   ├── pages/             # Multi-page dashboard components
│   └── app.py             # Dashboard entry point
├── notebooks/             # EDA and ML training pipelines
├── docker-compose.yml     # Container orchestration
└── requirements.txt       # Project dependencies
```

## 📊 Features
### 1. No-Show Prediction
Uses an **XGBoost** model trained on synthetic and historical booking data to calculate a probability score (0-1) for every passenger. Features include booking lead time, route demand, seasonality, and passenger demographics.

### 2. Revenue Simulation Dashboard
Interactive dashboard allowing users to:
- **Overview**: Track high-level KPIs like overall no-show rates and airline performance.
- **Simulate**: Compare "No Overbooking", "Fixed Overbooking", and "Dynamic ML Overbooking" strategies.
- **Analyze**: Visualize the trade-off between empty seats (wasted capacity) and bumped passengers (compensation costs).

## 🏃 Getting Started

### Using Docker (Recommended)
Launch the entire stack with one command:
```bash
docker-compose up --build
```
- **Frontend**: [http://localhost:8501](http://localhost:8501)
- **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Manual Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Backend**:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
3. **Run Frontend**:
   ```bash
   streamlit run frontend/app.py
   ```

## 📈 Business Insights
Dynamic overbooking consistently outperforms fixed strategies by:
- **Reducing Empty Seats**: Increasing utilization by selling exactly what the model predicts will be vacant.
- **Lowering Costs**: Minimizing expensive denied boarding compensations by avoiding aggressive overbooking on low-risk flights.
- **Maximizing Net Revenue**: Achieving a balanced "sweet spot" for profit.

---
Built with ❤️ for Airline Revenue Management.
