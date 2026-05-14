from pydantic import BaseModel
from typing import List, Optional

class PassengerData(BaseModel):
    Airline: str
    Departure_Airport: str
    Arrival_Airport: str
    Flight_Duration_Minutes: int
    Flight_Status: str
    Distance_Miles: int
    Price_USD: float
    Age: int
    Gender: str
    Income_Level: str
    Travel_Purpose: str
    Seat_Class: str
    Bags_Checked: int
    Frequent_Flyer_Status: str
    Check_in_Method: str
    Flight_Satisfaction_Score: float
    Delay_Minutes: float
    Booking_Days_In_Advance: int
    Weather_Impact: int
    Seat_Selected: str
    Route: str
    route_avg_passengers: float
    route_avg_purchase_lead: float
    route_avg_length_of_stay: float
    route_prop_extra_baggage: float
    route_prop_preferred_seat: float
    route_prop_in_flight_meals: float
    route_booking_conversion_rate: float
    Departure_Month: int
    Departure_DayOfWeek: int
    Departure_Hour: int
    Is_Weekend: int
    Route_Freq: float

class PredictionResponse(BaseModel):
    no_show_probability: float

class SimulationRequest(BaseModel):
    flight_capacity: int = 100
    ticket_price: float = 200.0
    compensation: float = 400.0
    passengers: List[PassengerData]

class SimulationResponse(BaseModel):
    strategy: str
    tickets_sold: int
    actual_show_ups: int
    revenue: float
    bumped: int
    compensation_cost: float
    empty_seats: int
    net_revenue: float
