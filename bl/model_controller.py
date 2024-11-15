import os
import numpy as np
import tensorflow as tf
from dl.stock_data import get_stock_data, process_data, create_dataset
from config.settings import model_dir, time_step, forecast_horizon

class ModelController:
    def train(self, ticker: str):
        prices = get_stock_data(ticker)
        scaled_prices, scaler = process_data(prices)
        X, Y = create_dataset(scaled_prices, time_step, forecast_horizon)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        model = self.create_model()

        model.fit(X, Y, epochs=100, batch_size=32)

        model_save_path = os.path.join(model_dir, f'{ticker}_model.h5')  
        scaler_save_path = os.path.join(model_dir, f'{ticker}_scaler.npy')
        
        model.save(model_save_path)  
        np.save(scaler_save_path, scaler)
        
        return f"Model trained and saved successfully for {ticker}"

    def predict(self, ticker: str):
        model_path = os.path.join(model_dir, f'{ticker}_model.h5')
        scaler_path = os.path.join(model_dir, f'{ticker}_scaler.npy')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise Exception("Model not found. Please train the model first.")
        
        model = tf.keras.models.load_model(model_path)
        scaler = np.load(scaler_path, allow_pickle=True).item()

        prices = get_stock_data(ticker)
        scaled_prices, _ = process_data(prices)
        X = np.array([scaled_prices[-time_step:]])
        X = X.reshape(X.shape[0], X.shape[1], 1)

        prediction = model.predict(X)
        prediction = scaler.inverse_transform(prediction)
        return prediction[0].tolist()

    def list_models(self):
        if not os.path.exists(model_dir):
            return []
        models = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
        return models

    def delete_model(self, ticker: str):
        model_path = os.path.join(model_dir, f'{ticker}_model.h5')
        scaler_path = os.path.join(model_dir, f'{ticker}_scaler.npy')

        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(scaler_path):
            os.remove(scaler_path)

        return f"Model and scaler for {ticker} deleted successfully"

    def check_model_status(self, ticker: str):
        model_path = os.path.join(model_dir, f'{ticker}_model.h5')
        if os.path.exists(model_path):
            return "Model exists and is ready"
        else:
            return "Model not found"

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(tf.keras.layers.LSTM(50, return_sequences=False))
        model.add(tf.keras.layers.Dense(forecast_horizon))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def update_settings(self, new_time_step, new_forecast_horizon):
        global time_step, forecast_horizon
        time_step = new_time_step
        forecast_horizon = new_forecast_horizon
        return f"Settings updated: time_step={time_step}, forecast_horizon={forecast_horizon}"
