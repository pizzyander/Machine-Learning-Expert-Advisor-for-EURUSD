import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import requests
import json
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Step 1: Connect to MT5 and Fetch Recent Candlestick Data
def fetch_recent_data(symbol, timeframe, n_candles):
    if not mt5.initialize():
        print("Failed to initialize MT5")
        quit()

    utc_from = datetime.now() - timedelta(days=1)  # Fetch last day's data
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, n_candles)
    mt5.shutdown()

    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data

symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_H3
n_candles = 60  # Increase to 60 candles to get 17 features per candle
data = fetch_recent_data(symbol, timeframe, n_candles)

# Step 2: Preprocess Data for LSTM Model
def preprocess_data(data):
    # Calculate technical indicators
    data['SMA_10'] = data['close'].rolling(window=10).mean()
    data['EMA_10'] = data['close'].ewm(span=10, adjust=False).mean()
    data['RSI'] = 100 - (100 / (1 + (data['close'].diff(1).where(data['close'].diff(1) > 0, 0).rolling(window=14).mean() / 
                                    -data['close'].diff(1).where(data['close'].diff(1) < 0, 0).rolling(window=14).mean())))
    data['MACD'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['BB_Middle'] = data['close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['close'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['close'].rolling(window=20).std()
    data['prev_close'] = data['close'].shift(1)
    data['ATR'] = data.apply(lambda row: max(
        row['high'] - row['low'], 
        abs(row['high'] - row['prev_close']), 
        abs(row['low'] - row['prev_close'])
    ), axis=1).rolling(window=14).mean()
    data['Stochastic'] = ((data['close'] - data['low'].rolling(window=14).min()) / 
                          (data['high'].rolling(window=14).max() - data['low'].rolling(window=14).min())) * 100
    data['Donchian_Upper'] = data['high'].rolling(window=20).max()
    data['Donchian_Lower'] = data['low'].rolling(window=20).min()
    data['Std_Dev'] = data['close'].rolling(window=14).std()
    data['CV'] = data['Std_Dev'] / data['close'].rolling(window=14).mean()
    data['ROC'] = (data['close'] - data['close'].shift(14)) / data['close'].shift(14) * 100
    data['Momentum'] = data['close'] - data['close'].shift(4)
    data['WilliamsR'] = ((data['high'].rolling(14).max() - data['close']) / (data['high'].rolling(14).max() - data['low'].rolling(14).min())) * -100

    # Fill NaN values
    data.fillna(data.mean(), inplace=True)

    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['close', 'SMA_10', 'EMA_10', 'RSI', 'MACD', 'Signal_Line', 
                                             'BB_Upper', 'BB_Lower', 'ATR', 'Stochastic', 'Donchian_Upper', 
                                             'Donchian_Lower', 'Std_Dev', 'ROC', 'Momentum', 'WilliamsR', 'CV']])
    
    print(f"Preprocessed Data Shape: {scaled_data.shape}")
    return scaled_data[-50:]  
 
# Step 3: Send Data to REST API for Prediction
def get_prediction(data):
    url = "http://127.0.0.1:8000/predict"  # Replace with your API endpoint
    headers = {'Content-Type': 'application/json'}
    
    payload = json.dumps({"features": data.tolist()})
    response = requests.post(url, headers=headers, data=payload)
    
    if response.status_code == 200:
        return response.json()['prediction']
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Step 4: Compare Prediction to Current Price and EMA Sequence
def decide_trade(prediction, current_price, ema_50, ema_200):
    ema_spread = ema_50 - ema_200
    if prediction > current_price and ema_spread > 0:  # Bullish signal
        return "Buy"
    elif prediction < current_price and ema_spread < 0:  # Bearish signal
        return "Sell"
    else:
        return "Hold"

# Step 5: Main Workflow
if __name__ == "__main__":
    scaled_data = preprocess_data(data)
    prediction = get_prediction(scaled_data)
    if prediction is None:
        print("Failed to get prediction.")
        exit()
    current_price = data['close'].iloc[-1]
    ema_50 = data['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    ema_200 = data['close'].ewm(span=200, adjust=False).mean().iloc[-1]
    decision = decide_trade(prediction, current_price, ema_50, ema_200)
    print(f"Prediction: {prediction}, Current Price: {current_price}, EMA 50-200 Spread: {ema_50 - ema_200}")
    print(f"Decision: {decision}")
