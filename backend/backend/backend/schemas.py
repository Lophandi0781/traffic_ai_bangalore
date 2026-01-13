from pydantic import BaseModel, Field
from datetime import datetime

class PredictRequest(BaseModel):
    location_id: str = Field(..., examples=["Silk Board Junction"])
    timestamp: datetime
    horizon_minutes: int = Field(30, ge=15, le=180)
    is_rain: int = Field(0, ge=0, le=1)
    is_event: int = Field(0, ge=0, le=1)

class PredictResponse(BaseModel):
    location_id: str
    timestamp: datetime
    horizon_minutes: int
    predicted_speed_kmph: float
    congestion_label: str
