import os
import asyncio
import json
import numpy as np
import pandas as pd
import requests
from metaapi_cloud_sdk import MetaApi
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
import aiocron

# Load credentials
with open('settings.json', 'r') as file:
    settings = json.load(file)

token = settings.get('metaapi_access_token')
account_id = settings.get('metaapi_accountid')
prediction_url = settings.get('prediction_url')  # FastAPI endpoint
symbol = os.getenv('SYMBOL') or 'EURUSD'
lot_size = 0.01
trailing_stop_pips = 15  # 15 pips trailing stop
domain = settings.get('domain') or 'agiliumtrade.agiliumtrade.ai'
TELEGRAM_BOT_TOKEN = settings.get('telegram_bot_token')
TELEGRAM_CHAT_ID = settings.get('telegram_chat_id')

# Load CSV file into a DataFrame
csv_path = "models/data.csv"
data = pd.read_csv(csv_path)

# Display the first few rows
print(data.tail())
print(data.describe())

def compute_indicators(data): 
    data['MACD'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data['BB_Middle'] = data['close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['close'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['close'].rolling(window=20).std()

    data['RSI'] = 100 - (100 / (1 + (data['close'].diff(1).where(data['close'].diff(1) > 0, 0).rolling(window=14).mean() /
                               -data['close'].diff(1).where(data['close'].diff(1) < 0, 0).rolling(window=14).mean())))

    data['ATR'] = data['close'].diff().abs().rolling(window=14).mean()
    data['Momentum'] = data['close'] - data['close'].shift(4)
    data['ROC'] = (data['close'] - data['close'].shift(14)) / data['close'].shift(14) * 100
    data['Stochastic'] = ((data['close'] - data['low'].rolling(window=14).min()) / 
                      (data['high'].rolling(window=14).max() - data['low'].rolling(window=14).min())) * 100
    data['WilliamsR'] = ((data['high'].rolling(14).max() - data['close']) / 
                     (data['high'].rolling(14).max() - data['low'].rolling(14).min())) * -100
    data['CCI'] = (data['close'] - data['close'].rolling(20).mean()) / (0.015 * data['close'].rolling(20).std())
    data['CV'] = data['close'].rolling(window=14).std() / data['close'].rolling(window=14).mean()
    data['Donchian_Upper'] = data['high'].rolling(window=20).max()
    data['Donchian_Lower'] = data['low'].rolling(window=20).min()
    data['Std_Dev'] = data['close'].rolling(window=14).std()
    data['OBV'] = (np.sign(data['close'].diff()) * data['tickVolume']).cumsum()
    data['ADL'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']) * data['tickVolume']

    return data  # Return the modified DataFrame with indicators added

import matplotlib.pyplot as plt

def plot_recent_closes(data):
    """Plots the close prices of the most recent 30 candles."""
    
    recent_data = data.iloc[-30:]  # Get last 30 candles
    
    plt.figure(figsize=(10, 5))
    plt.plot(recent_data['time'], recent_data['close'], marker='o', linestyle='-', color='b', label='Close Price')
    
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.title("Recent 30 Candles - Close Price Over Time")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.legend()
    plt.grid(True)
    
    plt.show()

def scale_features(data):
    # Compute and add indicators to data
    data = compute_indicators(data)

    # Drop time and close columns
    feature_columns = data.columns.difference(["time", "close"])

    # Extract only the necessary features
    features = data[feature_columns]

    # Fill NaN values with column means
    features.fillna(features.mean(), inplace=True)

    # Load the scaler
    base_dir = "c:/Users/hp/Machine-Learning-Expert-Advisor-for-EURUSD"
    scaler_path = os.path.join(base_dir, "models", "X_scaler.pkl")

    if os.path.exists(scaler_path):
        X_scaler = joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Ensure the model is trained and saved.")

    X = features.values  # Convert DataFrame to NumPy array
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    scaled_data = X_scaler.transform(X)

    # Apply random masking to part of the data
    mask_prob = 0.1
    mask = np.random.rand(*scaled_data.shape) < mask_prob
    scaled_data[:, 1:3] = np.where(mask[:, 1:3], 0, scaled_data[:, 1:3])

    # Slice the last 30 observations
    scaled_data = scaled_data[-300:]
    return scaled_data.reshape(1, 30, -1)  # Ensure correct shape


scaled_data = scale_features(data)
print("Scaled features shape:", scaled_data.shape)

def send_to_fastapi(scaled_data):
    """Sends the scaled data to FastAPI endpoint and gets prediction."""
    print("Scaled features shape before sending:", scaled_data.shape)
    response = requests.post(prediction_url, json={'features': scaled_data.tolist()})
    print("FastAPI Response:", response.json())  # Debugging line
    return response.json().get('prediction')

def send_telegram_message(message): 
    """Send a message to a Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            logging.info("Telegram message sent successfully.")
        else:
            logging.error(f"Failed to send Telegram message: {response.status_code}")
    except Exception as e:
        logging.error(f"Error sending Telegram message: {str(e)}")


async def main():
    """Main function to execute trading logic."""
    print(f"Running trade execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load CSV file into a DataFrame inside main
    csv_path = "models/data.csv"
    data = pd.read_csv(csv_path)  # Load fresh data

    # Compute indicators
    data = compute_indicators(data)
    
    print(f"Data shape after computing indicators: {data.shape}")

    # Ensure there are at least 30 rows before scaling
    if len(data) < 30:
        print(f"Insufficient data: Expected at least 30 rows, got {len(data)}")
        return

    # Scale the features
    scaled_data = scale_features(data.iloc[-30:])

    print("Scaled features shape:", scaled_data.shape)

    # Send data to FastAPI for prediction
    predicted_price = send_to_fastapi(scaled_data)
    print("Predicted price:", predicted_price)
    plot_recent_closes(data)

asyncio.run(main())
