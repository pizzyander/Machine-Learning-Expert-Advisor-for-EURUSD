import asyncio
import json
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from metaapi_cloud_sdk import MetaApi
import os

# Define the storage directory inside a mounted volume
MODELS_DIR = os.path.abspath("models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load credentials from a mounted settings.json file
settings_path = settings_path = "settings.json"
with open(settings_path, 'r') as file:
    settings = json.load(file)

token = settings.get('metaapi_access_token')
account_id = settings.get('metaapi_accountid')
symbol = 'EURUSD'
domain = settings.get('domain') or 'agiliumtrade.agiliumtrade.ai'

async def retrieve_historical_candles():
    api = MetaApi(token, {'domain': domain})
    account = await api.metatrader_account_api.get_account(account_id)
    
    if account.state != 'DEPLOYED':
        await account.deploy()
    if account.connection_status != 'CONNECTED':
        await account.wait_connected()
    
    num_candles = 20000
    start_time = datetime.now(timezone.utc)
    candles = []
    
    while len(candles) < num_candles:
        new_candles = await account.get_historical_candles(symbol, '4h', start_time)
        if not new_candles:
            break
        candles.extend(new_candles)
        start_time = new_candles[0]['time'] - timedelta(hours=4)
        if len(candles) >= num_candles:
            candles = candles[:num_candles]
            break

    df = pd.DataFrame(candles)
    
    if not df.empty:
        df = df[['time', 'open', 'high', 'low', 'close', 'tickVolume']]
        df['time'] = pd.to_datetime(df['time'])
    return df

# Load and preprocess data
data = asyncio.run(retrieve_historical_candles())
prices = data['close'].values

# Compute price changes and label encoding
price_changes = np.diff(prices)
price_states = ['Up' if change > 0.001 else 'Down' if change < -0.001 else 'Stable' for change in price_changes]
state_encoder = LabelEncoder()
state_encoded = state_encoder.fit_transform(price_states)

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

    indicators = {
        'MACD': data['MACD'],
        'Signal_Line': data['Signal_Line'],
        'RSI': data['RSI'],
        'BB_Middle': data['BB_Middle'],
        'BB_Upper': data['BB_Upper'],
        'BB_Lower': data['BB_Lower'],
        'ATR': data['ATR'],
        'Momentum': data['Momentum'],
        'ROC': data['ROC'],
        'Stochastic': data['Stochastic'],
        'WilliamsR': data['WilliamsR'],
        'CCI': data['CCI'],
        'CV': data['CV'],
        'Donchian_Upper': data['Donchian_Upper'],
        'Donchian_Lower': data['Donchian_Lower'],
        'Std_Dev': data['Std_Dev'],
        'OBV': data['OBV'],
        'ADL': data['ADL']
        }

    # List of indicators
    indicators = list(indicators.keys())
    return indicators

# Prepare data for training
def prepare_data(data):
    indicators = compute_indicators(data)
    data.dropna(inplace=True)

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X = X_scaler.fit_transform(data[indicators])
    y = data['close'].shift(-3).dropna().values.reshape(-1, 1)
    y_scaled = y_scaler.fit_transform(y)

    with open(os.path.join(MODELS_DIR, "X_scaler.pkl"), "wb") as f:
        pickle.dump(X_scaler, f)
    with open(os.path.join(MODELS_DIR, "y_scaler.pkl"), "wb") as f:
        pickle.dump(y_scaler, f)

    X_seq, y_seq = [], []
    sequence_length = 30
    for i in range(sequence_length, len(X) - 3):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y_scaled[i][0])
    return np.array(X_seq), np.array(y_seq)

X, y = prepare_data(data)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

import os
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define the model-building function for hyperparameter tuning
def build_model(hp):
    model = Sequential()

    # First GRU layer
    model.add(GRU(hp.Int('gru_units_1', min_value=64, max_value=256, step=64), 
                  return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(BatchNormalization())

    # Second GRU layer
    model.add(GRU(hp.Int('gru_units_2', min_value=64, max_value=256, step=64), return_sequences=True))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(BatchNormalization())

    # Third GRU layer
    model.add(GRU(hp.Int('gru_units_3', min_value=32, max_value=128, step=32), return_sequences=True))
    model.add(Dropout(hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(BatchNormalization())

    # Fourth GRU layer
    model.add(GRU(hp.Int('gru_units_4', min_value=32, max_value=128, step=32)))
    model.add(Dropout(hp.Float('dropout_4', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(BatchNormalization())

    # Fully connected layers
    model.add(Dense(hp.Int('dense_units', min_value=32, max_value=128, step=32)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# Hyperparameter search
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=10,  # Number of hyperparameter combinations to try
    executions_per_trial=1,
    directory='hyperparameter_tuning',
    project_name='GRU_Tuning'
)

# Search for best hyperparameters
tuner.search(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val), 
             callbacks=[EarlyStopping(monitor="val_loss", patience=5)])

# Get best hyperparameters and train final model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

# Callbacks for training
early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(X_train, y_train, epochs=100, batch_size=16, 
                    validation_data=(X_val, y_val), callbacks=[early_stop, reduce_lr])

# Save model after training
model_save_path = os.path.join(MODELS_DIR, "gru_model.keras")
model.save(model_save_path)
print(f"Optimized Model saved at {model_save_path}.")


# Evaluate the model
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, history):
    with open(os.path.join(MODELS_DIR, "y_scaler.pkl"), "rb") as f:
        y_scaler = pickle.load(f)

    y_pred = model.predict(X_test)
    y_pred_inversed = y_scaler.inverse_transform(y_pred)
    y_test_inversed = y_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Print the most recent 20 actual and predicted prices
    print("Most recent 20 actual prices:")
    print(y_test_inversed[-20:].flatten())

    print("\nMost recent 20 predicted prices:")
    print(y_pred_inversed[-20:].flatten())

    # Plot predictions vs actual values
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_inversed, label="Actual Prices", color='blue', alpha=0.7)
    plt.plot(y_pred_inversed, label="Predicted Prices", color='red', linestyle="dashed", alpha=0.7)
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.title("Predicted vs. Actual Prices")
    plt.legend()
    plt.grid(True)
    plt.show()

    return y_pred_inversed, y_test_inversed

# Call evaluation function
y_pred_inversed, y_test_inversed = evaluate_model(model, X_test, y_test, history)
