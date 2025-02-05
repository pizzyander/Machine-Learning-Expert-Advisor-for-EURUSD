import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import time
import logging
import datetime
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Logging configuration
log_file = "strategy1_mt5_model.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

# Constants
MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
USERNAME = 9749492
PASSWORD = "Abiola123@"
SERVER = "DerivSVG-Server"
lot_size = 0.1
PIP_VALUES = {
    "EURUSD": 0.0001,
    # Add other symbols as needed
}
TRAILING_STOP_PIPS = 12.5  # Trailing stop in pips
TAKE_PROFIT_MULTIPLIER = 1.0  # Take profit at the predicted price

# For our LSTM model (should match training settings)
time_window = 50  # Lookback window used for model training

# ---------------------------
# Step 1: Fetch Historical Data from MT5
# ---------------------------
def fetch_historical_data(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data

# ---------------------------
# Step 2: Preprocess Data (Feature Engineering and Scaling)
# ---------------------------
def preprocess_data(data):
    # Calculate technical indicators
    data['SMA_10'] = data['close'].rolling(window=10).mean()
    data['EMA_10'] = data['close'].ewm(span=10, adjust=False).mean()
    
    # RSI Calculation
    delta = data['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    data['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    # MACD and Signal Line
    data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    data['BB_Middle'] = data['close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['close'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['close'].rolling(window=20).std()
    
    # ATR
    data['prev_close'] = data['close'].shift(1)
    data['High_Low'] = data['high'] - data['low']
    data['High_Close'] = abs(data['high'] - data['prev_close'])
    data['Low_Close'] = abs(data['low'] - data['prev_close'])
    data['TR'] = data[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    data['ATR'] = data['TR'].rolling(window=14).mean()
    
    # Stochastic Oscillator
    data['Lowest_Low'] = data['low'].rolling(window=14).min()
    data['Highest_High'] = data['high'].rolling(window=14).max()
    data['Stochastic'] = ((data['close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low'])) * 100
    
    # Donchian Channels
    data['Donchian_Upper'] = data['high'].rolling(window=20).max()
    data['Donchian_Lower'] = data['low'].rolling(window=20).min()
    
    # Standard Deviation, CV, ROC
    data['Std_Dev'] = data['close'].rolling(window=14).std()
    data['CV'] = data['Std_Dev'] / data['close'].rolling(window=14).mean()
    data['ROC'] = (data['close'] - data['close'].shift(14)) / data['close'].shift(14) * 100
    
    # Williams %R and Short-Term Price Change
    data['Williams_%R'] = ((data['Highest_High'] - data['close']) / (data['Highest_High'] - data['Lowest_Low'])) * -100
    data['Price_Change_5'] = data['close'].diff(5)
    
    # Fill missing values
    data.fillna(method='bfill', inplace=True)
    
    # Select features to scale
    feature_cols = [
        'close', 'SMA_10', 'EMA_10', 'RSI', 'MACD', 'Signal_Line',
        'BB_Upper', 'BB_Lower', 'ATR', 'Stochastic',
        'Donchian_Upper', 'Donchian_Lower', 'Std_Dev', 'ROC',
        'Williams_%R', 'Price_Change_5'
    ]
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[feature_cols])
    print(f"Preprocessed Data Shape: {scaled_data.shape}")
    return scaled_data, scaler

# ---------------------------
# Step 3: Create Time-Series Sequences for LSTM Inference
# ---------------------------
def create_sequences(data, time_window):
    X_seq = []
    for i in range(len(data) - time_window):
        X_seq.append(data[i:i + time_window])
    return np.array(X_seq)

# ---------------------------
# Step 4: Load Trained Model and Scaler
# ---------------------------
model = load_model('forex_lstm_model.h5')
price_scaler = joblib.load('price_scaler.pkl')

# ---------------------------
# Step 5: Simulate Trading with Trailing Stop and Take Profit
# ---------------------------
def simulate_trading(data, predictions, trailing_stop_pips, take_profit_multiplier):
    positions = []  # Track open positions
    trades = []  # Track completed trades
    balance = 10000  # Starting balance
    balance_history = [balance]
    
    for i in range(len(predictions)):
        current_price = data['close'].iloc[i + time_window]
        predicted_price = predictions[i][0]
        
        # Check for take profit or trailing stop on open positions
        for position in positions[:]:
            entry_price, stop_loss, take_profit, direction = position
            if direction == "Buy":
                # Update trailing stop
                new_stop_loss = current_price - trailing_stop_pips * PIP_VALUES["EURUSD"]
                stop_loss = max(stop_loss, new_stop_loss)
                
                # Check for take profit or stop loss
                if current_price >= take_profit or current_price <= stop_loss:
                    pnl = (current_price - entry_price) * lot_size * 100000  # PnL in account currency
                    balance += pnl
                    trades.append({
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "direction": direction,
                        "pnl": pnl
                    })
                    positions.remove(position)
            elif direction == "Sell":
                # Update trailing stop
                new_stop_loss = current_price + trailing_stop_pips * PIP_VALUES["EURUSD"]
                stop_loss = min(stop_loss, new_stop_loss)
                
                # Check for take profit or stop loss
                if current_price <= take_profit or current_price >= stop_loss:
                    pnl = (entry_price - current_price) * lot_size * 100000  # PnL in account currency
                    balance += pnl
                    trades.append({
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "direction": direction,
                        "pnl": pnl
                    })
                    positions.remove(position)
        
        # Open new positions based on predictions
        if predicted_price > current_price:  # Buy signal
            take_profit = predicted_price * take_profit_multiplier
            stop_loss = current_price - trailing_stop_pips * PIP_VALUES["EURUSD"]
            positions.append((current_price, stop_loss, take_profit, "Buy"))
        elif predicted_price < current_price:  # Sell signal
            take_profit = predicted_price * take_profit_multiplier
            stop_loss = current_price + trailing_stop_pips * PIP_VALUES["EURUSD"]
            positions.append((current_price, stop_loss, take_profit, "Sell"))
        
        # Record balance history
        balance_history.append(balance)
    
    return trades, balance_history

# ---------------------------
# Step 6: Main Workflow
# ---------------------------
def main():
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_H3
    start_date = datetime(2024, 1, 1)
    end_date = datetime.now()
    
    # Fetch historical data
    data_raw = fetch_historical_data(symbol, timeframe, start_date, end_date)
    
    # Preprocess data and get scaled features
    scaled_data, feature_scaler = preprocess_data(data_raw)
    
    # Create sequences for model input
    X_seq = create_sequences(scaled_data, time_window)
    print(f"Sequence Data Shape: {X_seq.shape}")
    
    # Predict price changes using the LSTM model
    predictions = model.predict(X_seq)
    
    # Inverse transform the predictions using the price scaler
    scaled_predictions = price_scaler.inverse_transform(predictions)
    
    # Simulate trading with trailing stop and take profit
    trades, balance_history = simulate_trading(data_raw, scaled_predictions, TRAILING_STOP_PIPS, TAKE_PROFIT_MULTIPLIER)
    
    # Print trading results
    print(f"Total Trades: {len(trades)}")
    print(f"Final Balance: {balance_history[-1]:.2f}")
    
    # Plot balance history
    plt.figure(figsize=(12, 6))
    plt.plot(balance_history, label="Balance")
    plt.title("Balance Over Time")
    plt.xlabel("Time")
    plt.ylabel("Balance")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()