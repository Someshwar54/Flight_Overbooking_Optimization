from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine, Base
from .routes import api

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Flight Overbooking API")

# Configure CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Flight Overbooking API is running"}
