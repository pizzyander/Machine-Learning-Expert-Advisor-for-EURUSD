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
# Define the storage directory inside a mounted volume
MODELS_DIR = os.path.abspath("models")
os.makedirs(MODELS_DIR, exist_ok=True)

# File path
csv_file_path = os.path.join(MODELS_DIR, "data.csv1")

async def retrieve_historical_candles():
    """Fetches the most recent historical candles using MetaAPI"""
    
    api = MetaApi(token, {'domain': domain})
    account = await api.metatrader_account_api.get_account(account_id)

    if account.state != 'DEPLOYED':
        await account.deploy()
    if account.connection_status != 'CONNECTED':
        await account.wait_connected()

    num_candles = 10000  # Adjust as needed
    start_time = datetime.now(timezone.utc)  # Start fetching from the latest time

    # Fetch candles
    candles = await account.get_historical_candles(symbol, '4h', start_time)

    if not candles:
        print("No candles retrieved. Check your connection or symbol.")
        return pd.DataFrame()

    # Convert to DataFrame
    data = pd.DataFrame(candles)

    if not data.empty:
        data = data[['time', 'open', 'high', 'low', 'close', 'tickVolume']]
        data['time'] = pd.to_datetime(data['time'])  # Ensure proper datetime format
        data.sort_values('time', ascending=True, inplace=True)  # Ensure proper order
        # Define CSV file path
        csv_path = os.path.join("models", "data.csv1")
        # Save DataFrame to CSV
        data.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")

    print(f"Retrieved {len(data)} candles.") 
    return data

def prepare_new_entry(data):

    # Sort DataFrame by 'time' in descending order
    data = data.sort_values(by='time', ascending=True)

    # Get last close price (for Open in new row)
    last_close_price = data.iloc[0]['close']

    # Compute mean for 'high', 'low', 'tickvolume' from last 2 observations
    mean_high = data.iloc[:2]['high'].mean()
    mean_low = data.iloc[:2]['low'].mean()
    mean_tickvolume = data.iloc[:2]['tickVolume'].mean()

    # Create a new empty row (NaN values)
    new_row = {col: np.nan for col in data.columns}

    # Fill in required values
    new_row['time'] = data.iloc[-1]['time'] + pd.Timedelta(hours=4)  # Assuming 4H timeframe
    new_row['open'] = last_close_price  # Last close → Open
    new_row['high'] = mean_high
    new_row['low'] = mean_low
    new_row['tickVolume'] = mean_tickvolume

    # Append the new row at the bottom
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

    # Display the updated DataFrame
    print(data.tail())  # Check last few rows
    print(data.columns)
    return data

def compute_indicators(data): 
    data['MACD'] = data['open'].ewm(span=12, adjust=False).mean() - data['open'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data['BB_Middle'] = data['open'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['open'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['open'].rolling(window=20).std()

    data['RSI'] = 100 - (100 / (1 + (data['open'].diff(1).where(data['open'].diff(1) > 0, 0).rolling(window=14).mean() /
                               -data['open'].diff(1).where(data['open'].diff(1) < 0, 0).rolling(window=14).mean())))

    data['ATR'] = data['open'].diff().abs().rolling(window=14).mean()
    data['Momentum'] = data['open'] - data['open'].shift(4)
    data['ROC'] = (data['open'] - data['open'].shift(14)) / data['open'].shift(14) * 100
    data['Stochastic'] = ((data['open'] - data['low'].rolling(window=14).min()) / 
                      (data['high'].rolling(window=14).max() - data['low'].rolling(window=14).min())) * 100
    data['WilliamsR'] = ((data['high'].rolling(14).max() - data['open']) / 
                     (data['high'].rolling(14).max() - data['low'].rolling(14).min())) * -100
    data['CCI'] = (data['open'] - data['open'].rolling(20).mean()) / (0.015 * data['open'].rolling(20).std())
    data['CV'] = data['open'].rolling(window=14).std() / data['open'].rolling(window=14).mean()
    data['Donchian_Upper'] = data['high'].rolling(window=20).max()
    data['Donchian_Lower'] = data['low'].rolling(window=20).min()
    data['Std_Dev'] = data['open'].rolling(window=14).std()
    data['OBV'] = (np.sign(data['open'].diff()) * data['tickVolume']).cumsum()
    data['ADL'] = ((data['open'] - data['low']) - (data['high'] - data['open'])) / (data['high'] - data['low']) * data['tickVolume']
    data.fillna(data.mean(), inplace=True)
    return data 

def scale_features(data):

    data.drop(columns=['close', 'time'], inplace=True)

    print(data.columns)

    # Use absolute path relative to the container/project directory
    base_dir = os.path.abspath(os.path.dirname(__file__))  # Gets current script dir
    scaler_path = os.path.join(base_dir, "..", "models", "X_scaler.pkl")
    scaler_path = os.path.abspath(scaler_path)  # Ensures full absolute path

    if os.path.exists(scaler_path):
        X_scaler = joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Make sure it exists inside the container.")

    X = data.values  # Convert DataFrame to NumPy array
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    print(f"X shape in trade.py: {X.shape}")

    scaled_data = X_scaler.transform(X)

    # Apply random masking to part of the data
    mask_prob = 0.1
    mask = np.random.rand(*scaled_data.shape) < mask_prob
    scaled_data[:, 1:3] = np.where(mask[:, 1:3], 0, scaled_data[:, 1:3])

    # Slice the last 30 observations
    scaled_data = scaled_data[-30:]
    return scaled_data.reshape(1, 30, -1)

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
    # Fetch data
    data = await retrieve_historical_candles()

    data = prepare_new_entry(data)

    # Compute indicators
    data = compute_indicators(data)

    scaled_data = scale_features(data.iloc[-30:])
    
    # Ensure there are at least 30 rows before scaling
    if len(data) < 30:
        print(f"Insufficient data: Expected at least 30 rows, got {len(data)}")
        return

    print("Scaled features shape:", scaled_data.shape)

    # Send data to FastAPI for prediction
    predicted_price = send_to_fastapi(scaled_data)
    print("Predicted price:", predicted_price)
   
asyncio.run(main())
