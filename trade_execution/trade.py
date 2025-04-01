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

    num_candles = 500  # Fetch 500 candles
    start_time = datetime.now(timezone.utc)  # Start from the latest available time
    candles = []

    while len(candles) < num_candles:
        new_candles = await account.get_historical_candles(symbol, '4h', start_time)
        if not new_candles:
            break  # Stop if no new candles are retrieved

        candles.extend(new_candles)
        start_time = pd.to_datetime(new_candles[-1]['time']) - timedelta(hours=4)  # Go back further

        if len(candles) >= num_candles:
            break  # Stop once we reach 500 candles

    # Convert to DataFrame
    data = pd.DataFrame(candles[:num_candles])  # Limit to 10,000 candles

    if not data.empty:
        data = data[['time', 'open', 'high', 'low', 'close', 'tickVolume']]
        data['time'] = pd.to_datetime(data['time'])  # Ensure correct datetime format

    print(f"Retrieved {len(data)} candles.")  # Debugging line
    
    # Ensure 'models' directory exists
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Save DataFrame as CSV
    csv_path = os.path.join(models_dir, 'data.csv')
    data.to_csv(csv_path, index=False)

    print(f"Data saved to {csv_path}")
    return data

# Load and preprocess data
data = asyncio.run(retrieve_historical_candles())

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

def scale_features(data):
 
   # Compute and add indicators to data
    data = compute_indicators(data)

    # Fill NaN values with column means
    data.fillna(data.mean(), inplace=True)

    #drop time column
    data = data.drop(columns=["time"])
    for col in data.columns:
        print(col)
    import os
    import joblib

    base_dir = "c:/Users/hp/Machine-Learning-Expert-Advisor-for-EURUSD"
    scaler_path = os.path.join(base_dir, "models", "X_scaler.pkl")

    if os.path.exists(scaler_path):
        X_scaler = joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Ensure the model is trained and saved.")

    
        # Select features (excluding 'close')
    features = [col for col in data.columns if col != 'close']
    
    X = data[features].values  # Convert to NumPy array
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    scaled_data = X_scaler.transform(X)

    # Apply random masking to part of the data
    mask_prob = 0.1
    mask = np.random.rand(*scaled_data.shape) < mask_prob
    scaled_data[:, 1:3] = np.where(mask[:, 1:3], 0, scaled_data[:, 1:3]) 
    
    # slice last 30 observations
    scaled_data = scaled_data[-300:]
    scaled_data = np.array(scaled_data)  # Convert list to NumPy array
      
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

# Set pip value for calculating stop loss
PIP_VALUE = 0.0001  # Adjust if using JPY pairs (0.01)

async def place_trade(predicted_price):
    """Places a trade with dynamic lot sizing where SL is 1% of balance and TP is 3x SL."""
    api = MetaApi(token)
    account = await api.metatrader_account_api.get_account(account_id)
    await account.deploy()
    await account.wait_connected()
    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()

    # Retrieve account balance
    account_info = await connection.get_account_information()
    balance = account_info['balance']

    # Retrieve latest market price
    latest_price = await connection.get_symbol_price(symbol)
    entry_price = latest_price['ask'] if predicted_price > latest_price['ask'] else latest_price['bid']
    trade_direction = 'buy' if predicted_price > latest_price['ask'] else 'sell'

    # Calculate Stop Loss (SL) in money terms
    risk_amount = balance * 0.01  # 1% risk of balance
    tick_value = 10  # Approximate tick value for most pairs (adjust based on broker)

    # Determine lot size so SL in pips equals risk amount
    lot_size = max(0.01, round(risk_amount / (sl_pips * tick_value), 2))  # Adjust for minimum lot size 0.01

    # Convert SL to price value
    sl_pips = risk_amount / (lot_size * tick_value)  # Stop loss in pips
    stop_loss = entry_price - sl_pips if trade_direction == 'buy' else entry_price + sl_pips

    # Take Profit (TP) is 3x Stop Loss
    tp_pips = sl_pips * 3
    take_profit = entry_price + tp_pips if trade_direction == 'buy' else entry_price - tp_pips

    # Define trailing stop loss parameters
    trailing_stop = {
        'distance': {
            'distance': sl_pips,
            'units': 'RELATIVE_PIPS'
        }
    }

    # Place trade
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
        f"Lot Size: {lot_size}\n"
        f"Stop Loss: {stop_loss:.5f} ({sl_pips:.2f} pips)\n"
        f"Take Profit: {take_profit:.5f} ({tp_pips:.2f} pips)\n"
        f"Risk: {risk_amount:.2f} ({1}% of balance)\n"
        f"Trailing Stop: {sl_pips:.2f} pips"
    )
    send_telegram_message(message)

    print(f'Trade executed: {trade_direction} at {entry_price}, SL: {stop_loss}, TP: {take_profit}, Lot: {lot_size}, Result: {result}')

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

    scaled_data = scale_features(data.iloc[-30:])

    print("Scaled features shape:", scaled_data.shape)

    predicted_price = send_to_fastapi(scaled_data)

    if predicted_price is not None:
        await place_trade(predicted_price)
    else:
        print("No valid prediction received.")

asyncio.run(main())
