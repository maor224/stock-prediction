from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config.settings import time_step, forecast_horizon
from bl.model_controller import ModelController
from dl.stock_data import get_stock_data
from common.models.model_settings import ModelSettings
from common.models.stock_request import StockRequest

app = FastAPI()

model_controller = ModelController()


@app.post("/train")
def train_model(request: StockRequest):
    try:
        message = model_controller.train(request.ticker)
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(request: StockRequest):
    try:
        predictions = model_controller.predict(request.ticker)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/models")
def list_models():
    return {"models": model_controller.list_models()}

@app.delete("/model/{ticker}")
def delete_model(ticker: str):
    return {"message": model_controller.delete_model(ticker)}

@app.get("/stock_data/{ticker}")
def get_raw_stock_data(ticker: str):
    try:
        hist = get_stock_data(ticker)
        return {"data": hist.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/settings")
def update_settings(settings: ModelSettings):
    return model_controller.update_settings(settings.time_step, settings.forecast_horizon)

@app.get("/model_status/{ticker}")
def check_model_status(ticker: str):
    return model_controller.check_model_status(ticker)
