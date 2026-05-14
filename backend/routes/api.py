from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from ..schemas import PassengerData, PredictionResponse, SimulationRequest, SimulationResponse
from ..services import ml_service
from ..database import get_db
from ..models import SimulationResult

router = APIRouter()

@router.post("/predict/no-show", response_model=PredictionResponse)
def predict_no_show(passenger: PassengerData):
    prob = ml_service.predict_no_show_prob(passenger)
    return {"no_show_probability": prob}

@router.post("/simulate/revenue", response_model=List[SimulationResponse])
def simulate_revenue(req: SimulationRequest, db: Session = Depends(get_db)):
    results = ml_service.run_simulation(
        req.flight_capacity, 
        req.ticket_price, 
        req.compensation, 
        req.passengers
    )
    
    # Save to DB
    for res in results:
        db_res = SimulationResult(
            strategy=res["strategy"],
            total_net_revenue=res["net_revenue"],
            total_bumped=res["bumped"],
            total_empty_seats=res["empty_seats"]
        )
        db.add(db_res)
    db.commit()
    
    return results

@router.get("/data/summary")
def get_data_summary():
    # Stub for summary statistics for frontend
    return {
        "total_flights": 1000,
        "avg_no_show_rate": 0.05,
        "avg_revenue_loss": 50000
    }
