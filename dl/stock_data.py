import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fastapi import HTTPException

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(period="max")
        if hist.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")
        return hist['Close'].values.reshape(-1, 1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {e}")

def process_data(prices):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    return scaled_prices, scaler

def create_dataset(data, time_step=1, forecast_horizon=1):
    X, Y = [], []
    for i in range(len(data) - time_step - forecast_horizon):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[(i + time_step):(i + time_step + forecast_horizon), 0])
    return np.array(X), np.array(Y)
