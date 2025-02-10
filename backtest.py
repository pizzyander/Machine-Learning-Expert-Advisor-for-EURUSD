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

# Load model scaler
scaler_path = "./models/X_scaler.pkl"
scaler = joblib.load(scaler_path)

# Backtest parameters
initial_balance = 3000
lot_size = 0.1
tick_value = 10  # Adjust based on broker specifications
trailing_stop_pips = 20
pip_value = 0.0001  # Adjust for JPY pairs

# Load credentials
with open('settings.json', 'r') as file:
    settings = json.load(file)

token = settings.get('metaapi_access_token')
account_id = settings.get('metaapi_accountid')
prediction_url = settings.get('prediction_url')  # FastAPI endpoint
symbol = os.getenv('SYMBOL') or 'EURUSD'
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

    num_candles = 500  # Increase this to get more data
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
    features = scaler.feature_names_in_
    data = data[features].fillna(0)
    scaled_data = scaler.transform(data[-30:])
    return scaled_data.reshape(1, 30, -1)

def send_to_fastapi(data):
    prediction_url = "http://localhost:8000/predict"
    response = requests.post(prediction_url, json={'features': data.tolist()})
    prediction = response.json().get('prediction')
    
    if prediction is None:
        print("Warning: Received invalid prediction from model! Proceeding with NaN prediction.")
        return np.nan
    
    return prediction

async def backtest():
    global initial_balance
    df = await retrieve_historical_candles()
    df = compute_indicators(df)
    trades = []
    balance = initial_balance
    
    for i in range(30, len(df)):
        scaled_features = scale_features(df.iloc[i-30:i])
        predicted_price = send_to_fastapi(scaled_features)
        
        entry_price = df.iloc[i]['close']
        sl_price = entry_price - (trailing_stop_pips * pip_value)
        tp_price = predicted_price if not np.isnan(predicted_price) else entry_price  # Use entry price if prediction is NaN
        
        trade_result = (tp_price - entry_price) / pip_value * lot_size * tick_value
        balance += trade_result
        
        trades.append({
            'time': df.iloc[i]['time'],
            'entry_price': entry_price,
            'stop_loss': sl_price,
            'take_profit': tp_price,
            'profit_loss': trade_result,
            'balance': balance
        })
        
        if trade_result < 0:
            print(f"Warning: Large loss detected! Trade result: {trade_result:.2f}")
    
    results = pd.DataFrame(trades)
    print(results.tail(10))
    print(f"Final Balance: ${balance:.2f}")
    return results

if __name__ == "__main__":
    asyncio.run(backtest())
