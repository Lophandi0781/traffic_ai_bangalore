from fastapi import FastAPI
from .schemas import PredictRequest, PredictResponse
from .model import TrafficModel


app = FastAPI(title="Bangalore Traffic AI Predictor", version="1.0")
tm = TrafficModel()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    speed = tm.predict_speed(
        location_id=req.location_id,
        ts=req.timestamp,
        horizon_minutes=req.horizon_minutes,
        is_rain=req.is_rain,
        is_event=req.is_event,
    )
    return PredictResponse(
        location_id=req.location_id,
        timestamp=req.timestamp,
        horizon_minutes=req.horizon_minutes,
        predicted_speed_kmph=round(speed, 2),
        congestion_label=tm.congestion_label(speed),
    )
