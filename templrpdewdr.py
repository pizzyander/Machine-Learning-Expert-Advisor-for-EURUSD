import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import optuna
import shap
import datetime
import math
import matplotlib.pyplot as plt

# Step 1: Connect to MT5 and Fetch Data
def fetch_data(symbol, timeframe, n_candles):
    if not mt5.initialize():
        print("Failed to initialize MT5")
        quit()
    
    utc_from = datetime.datetime.now() - datetime.timedelta(days=30)
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, n_candles)
    mt5.shutdown()

    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data

symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_H3
n_candles = 5000
data = fetch_data(symbol, timeframe, n_candles)
prices = data['close'].values
print(f"Fetched {len(prices)} rows of EUR/USD data.")

# Step 2: Calculate Price Changes and Define Markov States
price_changes = np.diff(prices)
price_states = []
threshold = 0.001

for change in price_changes:
    if change > threshold:
        price_states.append("Up")
    elif change < -threshold:
        price_states.append("Down")
    else:
        price_states.append("Stable")

state_encoder = LabelEncoder()
state_encoded = state_encoder.fit_transform(price_states)

n_states = len(set(price_states))
transition_matrix = np.zeros((n_states, n_states))
for i in range(len(state_encoded) - 1):
    current_state = state_encoded[i]
    next_state = state_encoded[i + 1]
    transition_matrix[current_state, next_state] += 1

transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

# Generate Markov Probabilities
markov_probs = []
for i in range(len(state_encoded) - 1):
    current_state = state_encoded[i]
    probs = transition_matrix[current_state]
    markov_probs.append(probs)

markov_probs = np.array(markov_probs)

# Step 3: Add Technical Indicators
data['SMA_10'] = data['close'].rolling(window=10).mean()
data['EMA_10'] = data['close'].ewm(span=10, adjust=False).mean()

delta = data['close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

data['BB_Middle'] = data['close'].rolling(window=20).mean()
data['BB_Upper'] = data['BB_Middle'] + 2 * data['close'].rolling(window=20).std()
data['BB_Lower'] = data['BB_Middle'] - 2 * data['close'].rolling(window=20).std()

# ATR
data['High_Low'] = data['high'] - data['low']
data['High_Close'] = abs(data['high'] - data['close'].shift(1))
data['Low_Close'] = abs(data['low'] - data['close'].shift(1))
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

# Drop NaN rows
data = data.dropna()
prices = data['close'].values

# Step 4: Preprocess Data
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
scaled_features = [
    scaler.fit_transform(data[col].values.reshape(-1, 1)) for col in [
        'EMA_10', 'SMA_10', 'RSI', 'MACD', 'Signal_Line',
        'BB_Upper', 'BB_Lower', 'ATR', 'Stochastic',
        'Donchian_Upper', 'Donchian_Lower', 'Std_Dev', 'ROC'
    ]
]
# Ensure all arrays have the same length
min_length = min(
    len(scaled_prices), len(markov_probs),
    *[len(feature) for feature in scaled_features]
)

# Truncate all arrays to the minimum length
scaled_prices = scaled_prices[:min_length]
markov_probs = markov_probs[:min_length]
scaled_features = [feature[:min_length] for feature in scaled_features]

# Combine all features
X = np.hstack([scaled_prices, markov_probs] + scaled_features)
y = price_changes[1:]

# Create LSTM time-series data
time_window = 50
X_lstm, y_lstm = [], []
for i in range(len(X) - time_window):
    X_lstm.append(X[i:i + time_window])
    y_lstm.append(y[i + time_window])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm).reshape(-1, 1)

# Step 5: TimeSeriesSplit for Validation
tscv = TimeSeriesSplit(n_splits=5)

# Step 5.5: Split data globally for final model training
train_size = int(0.8 * len(X_lstm))  # 80% of the data for training
X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]

def objective(trial):
    units_1 = trial.suggest_int('units_1', 30, 100)
    units_2 = trial.suggest_int('units_2', 30, 100)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_int('batch_size', 16, 128)

    for train_idx, test_idx in tscv.split(X_lstm):
        X_train, X_test = X_lstm[train_idx], X_lstm[test_idx]
        y_train, y_test = y_lstm[train_idx], y_lstm[test_idx]

        model = Sequential([
            LSTM(units_1, return_sequences=True, input_shape=(time_window, X.shape[1])),
            LSTM(units_2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=batch_size, verbose=0)
        loss = model.evaluate(X_test, y_test, verbose=0)
        return loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print(f"Best Parameters: {study.best_params}")


# Step 6: Train Final Model
units_1, units_2 = study.best_params['units_1'], study.best_params['units_2']
learning_rate, batch_size = study.best_params['learning_rate'], study.best_params['batch_size']

model = Sequential([
    LSTM(units_1, return_sequences=True, input_shape=(time_window, X.shape[1])),
    LSTM(units_2),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])


# Step 7: SHAP Explainability
# Reshape X_train and X_test to 2D for SHAP (flatten time series data)
X_train_2D = X_train.reshape(X_train.shape[0], -1)  # Flatten (samples, time_steps * features)
X_test_2D = X_test.reshape(X_test.shape[0], -1)

# Create SHAP explainer
explainer = shap.KernelExplainer(lambda x: model.predict(x.reshape(-1, time_window, X.shape[1])), X_train_2D[:100])

# Compute SHAP values for a subset of X_test
shap_values = explainer.shap_values(X_test_2D[:10])

# Plot SHAP summary
shap.summary_plot(shap_values, X_test_2D[:10])


# Step 8: Evaluate Model
predictions = model.predict(X_test)
scaled_predictions = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test)

rmse = math.sqrt(mean_squared_error(actual_values, scaled_predictions))
mape = mean_absolute_percentage_error(actual_values, scaled_predictions) * 100
print(f"\nModel Performance:\nRMSE: {rmse:.5f}\nMAPE: {mape:.2f}%")

# Plot Predictions
plt.figure(figsize=(12, 6))
plt.plot(actual_values, label='Actual Prices')
plt.plot(scaled_predictions, label='Predicted Prices')
plt.legend()
plt.show()
