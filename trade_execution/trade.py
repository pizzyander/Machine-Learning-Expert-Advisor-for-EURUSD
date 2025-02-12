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

async def retrieve_historical_candles():
    api = MetaApi(token, {'domain': domain})
    account = await api.metatrader_account_api.get_account(account_id)

    if account.state != 'DEPLOYED':
        await account.deploy()
    if account.connection_status != 'CONNECTED':
        await account.wait_connected()

    num_candles = 100  # Increase this to get more data
    start_time = datetime.now(timezone.utc)
    candles = []

    while len(candles) < num_candles:
        new_candles = await account.get_historical_candles(symbol, '4h', start_time)
        if not new_candles:
            break
        candles.extend(new_candles)
        start_time = new_candles[0]['time'] - timedelta(hours=4)
        if len(candles) >= num_candles:
            break

    data = pd.DataFrame(candles)

    if not data.empty:
        data = data[['time', 'open', 'high', 'low', 'close', 'tickVolume']]
        data['time'] = pd.to_datetime(data['time'])
    
    print(f"Retrieved {len(data)} candles.")  # Debugging line

    return data


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
    return data

def scale_features(data):
    """Scales the computed features using a pre-saved MinMaxScaler."""
    if len(data) < 30:
        raise ValueError(f"Insufficient data: Expected at least 30 rows, got {len(data)}")
    scaler = "/app/models/X_scaler.pkl"
    X_scaler = joblib.load(scaler)
    
    feature_columns = X_scaler.feature_names_in_
    
    # Fill NaN values with 0 to avoid issues
    data = data[feature_columns].fillna(0)
    
    # Ensure correct feature shape before transforming
    scaled_data = X_scaler.transform(data[-30:])
    
    return scaled_data.reshape(1, 30, -1)  # Allow dynamic feature count


def send_to_fastapi(data):
    """Sends the scaled data to FastAPI endpoint and gets prediction."""
    response = requests.post(prediction_url, json={'features': data.tolist()})
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




# Set pip value for calculating stop loss
PIP_VALUE = 0.0001  # Adjust if using JPY pairs (0.01)

async def place_trade(predicted_price):
    """Places a trade based on the predicted price using MetaAPI."""
    api = MetaApi(token)
    account = await api.metatrader_account_api.get_account(account_id)
    await account.deploy()
    await account.wait_connected()
    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()
    
    latest_price = await connection.get_symbol_price(symbol)
    entry_price = latest_price['ask'] if predicted_price > latest_price['ask'] else latest_price['bid']
    trade_direction = 'buy' if predicted_price > latest_price['ask'] else 'sell'
    
    # Calculate Stop Loss (SL) based on 17-pip trailing stop
    sl_pips = 17 * PIP_VALUE  # Adjust for JPY pairs if needed
    stop_loss = entry_price - sl_pips if trade_direction == 'buy' else entry_price + sl_pips
    
    # Take Profit (TP) is set 60 pips away from entry price
    tp_pips = 60 * PIP_VALUE
    take_profit = entry_price + tp_pips if trade_direction == 'buy' else entry_price - tp_pips
    
    # Define trailing stop loss parameters
    trailing_stop = {
        'distance': {
            'distance': 17,
            'units': 'RELATIVE_PIPS'
        }
    }  # 17 pips trailing stop
    
    # Place trade with trailing stop loss
    if trade_direction == 'buy':
        result = await connection.create_market_buy_order(
            symbol, lot_size, stop_loss, take_profit, 
            {'comment': 'GRU Prediction', 'trailingStopLoss': trailing_stop}
        )
    else:
        result = await connection.create_market_sell_order(
            symbol, lot_size, stop_loss, take_profit, 
            {'comment': 'GRU Prediction', 'trailingStopLoss': trailing_stop}
        )
    
    # Send Telegram Notification
    message = (
        f"\U0001F4E2 **New Trade Alert** \U0001F4E2\n"
        f"Symbol: {symbol}\n"
        f"Direction: {trade_direction.upper()}\n"
        f"Entry Price: {entry_price:.5f}\n"
        f"Stop Loss: {stop_loss:.5f}\n"
        f"Take Profit: {take_profit:.5f} (60 pips away)\n"
        f"Trailing Stop: 17 pips"
    )
    send_telegram_message(message)
    
    print(f'Trade executed: {trade_direction} at {entry_price}, SL: {stop_loss}, TP: {take_profit}, result: {result}')

async def main():
    """Main function to execute trading logic."""
    print(f"Running trade execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data = await retrieve_historical_candles()
    data = compute_indicators(data)
    
    print(f"Data shape after computing indicators: {data.shape}")

    # Ensure there are at least 30 rows before scaling
    if len(data) < 30:
        print(f"Insufficient data: Expected at least 30 rows, got {len(data)}")
        return

    scaled_features = scale_features(data.iloc[-30:])

    print("Scaled features shape:", scaled_features.shape)

    predicted_price = send_to_fastapi(scaled_features)

    if predicted_price is not None:
        await place_trade(predicted_price)
    else:
        print("No valid prediction received.")

# Schedule execution every 4 hours at 5 minutes past (11:05 PM, 3:05 AM, etc.)
aiocron.crontab("5 23,3,7,11,15,19 * * *", func=main)

# Keep the script running indefinitely
asyncio.get_event_loop().run_forever()