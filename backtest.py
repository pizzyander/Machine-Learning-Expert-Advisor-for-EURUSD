import numpy as np
import pandas as pd
import json
import joblib
import requests
from datetime import datetime, timedelta

# Load model scaler
scaler_path = "./models/X_scaler.pkl"
scaler = joblib.load(scaler_path)

# Backtest parameters
initial_balance = 3000
lot_size = 0.1
tick_value = 1  # Adjust based on broker specifications
trailing_stop_pips = 15
pip_value = 0.0001  # Adjust for JPY pairs

# Load historical data
def generate_historical_data(num_candles=500):
    np.random.seed(42)
    timestamps = [datetime.now() - timedelta(minutes=5 * i) for i in range(num_candles)]
    close_prices = np.cumsum(np.random.randn(num_candles) * 0.0005 + 1.1)
    high_prices = close_prices + np.random.rand(num_candles) * 0.0003
    low_prices = close_prices - np.random.rand(num_candles) * 0.0003
    open_prices = close_prices + np.random.randn(num_candles) * 0.0002
    volumes = np.random.randint(100, 1000, num_candles)
    
    data = pd.DataFrame({
        'time': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'tickVolume': volumes
    })
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
    return response.json().get('prediction')

def backtest():
    global initial_balance
    df = generate_historical_data()
    df = compute_indicators(df)
    trades = []
    balance = initial_balance
    
    for i in range(30, len(df)):
        scaled_features = scale_features(df.iloc[i-30:i])
        predicted_price = send_to_fastapi(scaled_features)
        
        if predicted_price:
            entry_price = df.iloc[i]['close']
            sl_price = entry_price - (trailing_stop_pips * pip_value)
            tp_price = predicted_price
            trade_result = (tp_price - entry_price) * (10 / pip_value) * lot_size
            balance += trade_result
            
            trades.append({
                'time': df.iloc[i]['time'],
                'entry_price': entry_price,
                'stop_loss': sl_price,
                'take_profit': tp_price,
                'profit_loss': trade_result,
                'balance': balance
            })
    
    results = pd.DataFrame(trades)
    print(results.tail(10))
    print(f"Final Balance: ${balance:.2f}")
    return results

if __name__ == "__main__":
    backtest()
