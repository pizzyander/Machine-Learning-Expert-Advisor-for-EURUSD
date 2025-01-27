import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import datetime
import math

# Step 1: Connect to MT5 and Fetch EUR/USD Data
def fetch_data(symbol, timeframe, n_candles):
    if not mt5.initialize():
        print("Failed to initialize MT5")
        quit()
    
    # Fetch data
    utc_from = datetime.datetime.now() - datetime.timedelta(days=30)  # Last 30 days of data
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, n_candles)
    mt5.shutdown()

    # Convert to DataFrame
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data

# Fetch hourly data for EUR/USD
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_H1
n_candles = 5000  # Fetch last 5000 candles
data = fetch_data(symbol, timeframe, n_candles)

prices = data['close'].values  # Closing prices
print(f"Fetched {len(prices)} rows of EUR/USD data.")

# Step 2: Calculate Price Changes and Define Markov States
price_changes = np.diff(prices)
price_states = []

# Define Markov states
threshold = 0.001
for change in price_changes:
    if change > threshold:
        price_states.append("Up")
    elif change < -threshold:
        price_states.append("Down")
    else:
        price_states.append("Stable")

# Encode states into numerical values for Markov Model
state_encoder = LabelEncoder()
state_encoded = state_encoder.fit_transform(price_states)

# Step 3: Build Markov Transition Matrix
n_states = len(set(price_states))
transition_matrix = np.zeros((n_states, n_states))

for i in range(len(state_encoded) - 1):
    current_state = state_encoded[i]
    next_state = state_encoded[i + 1]
    transition_matrix[current_state, next_state] += 1

# Normalize rows to get probabilities
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
print("\nMarkov Transition Matrix (EUR/USD):")
print(pd.DataFrame(transition_matrix, columns=state_encoder.classes_, index=state_encoder.classes_))

# Generate Markov Probabilities
markov_probs = []
predicted_markov_states = []
for i in range(len(state_encoded) - 1):
    current_state = state_encoded[i]
    probs = transition_matrix[current_state]
    markov_probs.append(probs)
    predicted_markov_states.append(np.argmax(probs))  # Most probable state

markov_probs = np.array(markov_probs)

# Step 4: Calculate EMA, SMA, RSI, MACD, and Bollinger Bands
# EMA and SMA
data['SMA_10'] = data['close'].rolling(window=10).mean()  # Simple Moving Average (10-period)
data['EMA_10'] = data['close'].ewm(span=10, adjust=False).mean()  # Exponential Moving Average (10-period)

# RSI (Relative Strength Index)
delta = data['close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# MACD (Moving Average Convergence Divergence)
data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands
data['BB_Middle'] = data['close'].rolling(window=20).mean()
data['BB_Upper'] = data['BB_Middle'] + 2 * data['close'].rolling(window=20).std()
data['BB_Lower'] = data['BB_Middle'] - 2 * data['close'].rolling(window=20).std()

data = data.dropna()  # Drop rows with NaN values (first few rows due to rolling calculations)
prices = data['close'].values  # Update prices after dropping NaN rows

# Step 5: Preprocess Data for LSTM
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices[:-1].reshape(-1, 1))  # Normalize prices
ema_scaled = scaler.fit_transform(data['EMA_10'].values.reshape(-1, 1))
sma_scaled = scaler.fit_transform(data['SMA_10'].values.reshape(-1, 1))
rsi_scaled = scaler.fit_transform(data['RSI'].values.reshape(-1, 1))
macd_scaled = scaler.fit_transform(data['MACD'].values.reshape(-1, 1))
signal_scaled = scaler.fit_transform(data['Signal_Line'].values.reshape(-1, 1))
bb_upper_scaled = scaler.fit_transform(data['BB_Upper'].values.reshape(-1, 1))
bb_lower_scaled = scaler.fit_transform(data['BB_Lower'].values.reshape(-1, 1))

# Ensure all arrays have the same length
min_length = min(
    len(scaled_prices), len(markov_probs),
    len(ema_scaled), len(sma_scaled),
    len(rsi_scaled), len(macd_scaled),
    len(signal_scaled), len(bb_upper_scaled),
    len(bb_lower_scaled)
)

# Truncate all arrays to the minimum length
scaled_prices = scaled_prices[:min_length]
markov_probs = markov_probs[:min_length]
ema_scaled = ema_scaled[:min_length]
sma_scaled = sma_scaled[:min_length]
rsi_scaled = rsi_scaled[:min_length]
macd_scaled = macd_scaled[:min_length]
signal_scaled = signal_scaled[:min_length]
bb_upper_scaled = bb_upper_scaled[:min_length]
bb_lower_scaled = bb_lower_scaled[:min_length]

# Combine all features
X = np.hstack([
    scaled_prices,
    markov_probs,
    ema_scaled,
    sma_scaled,
    rsi_scaled,
    macd_scaled,
    signal_scaled,
    bb_upper_scaled,
    bb_lower_scaled
])

y = price_changes[1:]  # Target: price changes for regression

# Prepare LSTM input (reshape for time steps)
time_window = 10
X_lstm = []
y_lstm = []
for i in range(len(X) - time_window):
    X_lstm.append(X[i:i + time_window])
    y_lstm.append(y[i + time_window])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm).reshape(-1, 1)

# Split into training and testing sets
train_size = int(0.8 * len(X_lstm))
X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]

# Step 6: Build and Train the LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_window, X.shape[1])),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train the LSTM
print("\nTraining the LSTM model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Step 7: Evaluate and Predict
predictions = model.predict(X_test)
scaled_predictions = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test)

# RMSE and MAPE Evaluation
rmse = math.sqrt(mean_squared_error(actual_values, scaled_predictions))
mape = mean_absolute_percentage_error(actual_values, scaled_predictions) * 100

print("\nModel Performance:")
print(f"RMSE: {rmse:.5f}")
print(f"MAPE: {mape:.2f}%")

# Step 8: Output Markov States and Probabilities
print("\nPredicted Markov States and Probabilities:")
for i in range(10):  # Display the first 10 states and probabilities
    state = state_encoder.inverse_transform([predicted_markov_states[i]])[0]
    probs = markov_probs[i]
    print(f"State: {state}, Probabilities: {probs}")

# Compare predictions and actual values
print("\nPredicted vs Actual (EUR/USD):")
for pred, actual in zip(scaled_predictions[:10], actual_values[:10]):
    print(f"Predicted: {pred[0]:.5f}, Actual: {actual[0]:.5f}")
