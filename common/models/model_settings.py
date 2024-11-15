from pydantic import BaseModel

class ModelSettings(BaseModel):
    time_step: int
    forecast_horizon: int