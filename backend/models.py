from sqlalchemy import Column, Integer, String, Float, DateTime
import datetime
from .database import Base

class SimulationResult(Base):
    __tablename__ = "simulation_results"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    strategy = Column(String, index=True)
    total_net_revenue = Column(Float)
    total_bumped = Column(Integer)
    total_empty_seats = Column(Integer)
