import pandas as pd
import numpy as np
import logging
import traceback
from tensorflow.keras.models import load_model
from joblib import load
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load model and scalers
try:
    model = load_model("gru_model.h5")
    scaler_X = load("X_train_scaled.joblib")
    scaler_y = load("y_train_scaled.joblib")
    logging.info("Model and scalers loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model/scalers: {e}")
    raise RuntimeError("Failed to load model or scalers. Please check the file paths.")

# Define the FastAPI app
app = FastAPI()

# Define a request body model
class PredictionRequest(BaseModel):
    features: list  # Input features in JSON format (list of lists)

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Log the incoming request
        logging.info(f"Received request with data: {request.features}")

        # Convert features to a DataFrame, dropping unwanted columns if needed
        raw_data = pd.DataFrame(request.features)
        
        # Retain only open, high, low columns (drop other columns like 'close')
        data = raw_data[['open', 'high', 'low']]

        # Ensure data is numeric
        data = data.apply(pd.to_numeric, errors="coerce")

        # Feature engineering function
        def add_features(data):
            # Moving Averages
            data['SMA_20'] = data['open'].rolling(window=20).mean()
            data['EMA_50'] = data['open'].ewm(span=50, adjust=False).mean()
            # RSI
            delta = data['open'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['RSI_14'] = 100 - (100 / (1 + rs))
            # MACD
            data['MACD'] = data['open'].ewm(span=12, adjust=False).mean() - data['open'].ewm(span=26, adjust=False).mean()
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            # Volatility
            data['volatility_20'] = data['open'].pct_change().rolling(window=20).std()
            return data

        # Apply feature engineering
        data = add_features(data)

        # Drop rows with NaN values
        data.dropna(inplace=True)

        # Validate data after feature engineering
        if data.shape[0] < 90:
            raise ValueError("Insufficient data after feature engineering. At least 90 rows are required.")

        # Scale the incoming features
        scaled_data = scaler_X.transform(data)

        # Ensure we are using the last 90 time steps
        time_steps = 90
        if scaled_data.shape[0] < time_steps:
            raise ValueError(f"Not enough rows for prediction. Expected at least {time_steps}, got {scaled_data.shape[0]}.")
        
        #convert data to np array and ready for 
        np.array(data)

        scaled_data = scaled_data[-time_steps:]

        # Reshape for model input
        reshaped_data = scaled_data.reshape((1, time_steps, scaled_data.shape[1]))

        # Make prediction and inverse transform
        prediction_scaled = model.predict(reshaped_data)
        prediction = scaler_y.inverse_transform(prediction_scaled)

        # Log and return the prediction
        logging.info(f"Prediction result: {prediction.flatten().tolist()}")
        return {"prediction": prediction.flatten().tolist()}

    except Exception as e:
        # Log the error and return an HTTPException
        logging.error(f"Error occurred: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

