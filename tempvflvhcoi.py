import os
import asyncio
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Bidirectional, GRU, Dropout, Dense, Input, Multiply, Permute, Lambda
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from metaapi_cloud_sdk import MetaApi

# Load credentials from settings.json
with open('settings.json', 'r') as file:
    settings = json.load(file)

token = settings.get('metaapi_access_token')
account_id = settings.get('metaapi_accountid')
symbol = os.getenv('SYMBOL') or 'EURUSD'
domain = settings.get('domain') or 'agiliumtrade.agiliumtrade.ai'

async def retrieve_historical_candles():
    api = MetaApi(token, {'domain': domain})
    account = await api.metatrader_account_api.get_account(account_id)
    if account.state != 'DEPLOYED':
        await account.deploy()
    if account.connection_status != 'CONNECTED':
        await account.wait_connected()
    
    num_candles = 15000
    start_time = datetime.now(timezone.utc)
    candles = []
    while len(candles) < num_candles:
        new_candles = await account.get_historical_candles(symbol, '4h', start_time)
        if not new_candles:
            break
        candles.extend(new_candles)
        start_time = new_candles[0]['time'] - timedelta(hours=3)
        if len(candles) >= num_candles:
            candles = candles[:num_candles]
            break
    return pd.DataFrame(candles)

data = asyncio.run(retrieve_historical_candles())
data['time'] = pd.to_datetime(data['time'])
prices = data['close'].values

# Compute price changes
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
    data['OBV'] = (np.sign(data['close'].diff()) * data['volume']).cumsum()
    data['ADL'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']) * data['volume']

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
    indicators = list(indicators.keys())
    return indicators

scaler = MinMaxScaler()
indicators = compute_indicators(data)
scaled_features = scaler.fit_transform(data[indicators].dropna())

min_length = min(len(scaled_features) - 1, len(price_changes[1:]))

X_train, X_test, y_train, y_test = train_test_split(
    scaled_features[:min_length],
    price_changes[1:min_length+1],
    test_size=0.2, 
    shuffle=False
)

# Reshape data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Attention Layer
def attention_block(inputs):
    attention = Dense(inputs.shape[2], activation='softmax')(inputs)
    attention = Permute((2, 1))(attention)
    attention = Lambda(lambda x: K.mean(x, axis=1), name='attention_weights')(attention)
    output = Multiply()([inputs, attention[:, :, None]])
    return output

# Define Model
input_layer = Input(shape=(X_train.shape[1], 1))
x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=l2(0.001)))(x)
x = Bidirectional(GRU(64, return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=l2(0.001)))(x)
x = Bidirectional(GRU(32, return_sequences=True, recurrent_dropout=0.2))(x)
x = attention_block(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
output_layer = Dense(1, activation='linear')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse', metrics=['mse', 'mae'])

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test), verbose=1, shuffle=False)
model.save('forex_gru_model.h5')
joblib.dump(scaler, 'scaler.pkl')

# Predictions
y_pred = model.predict(X_test)
print("Actual vs Predicted Prices:")
for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual:.5f}, Predicted: {float(predicted[0]):.5f}")

# Plot Results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Price Changes', color='blue')
plt.plot(y_pred, label='Predicted Price Changes', color='red', linestyle='dashed')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price Change')
plt.title('Actual vs Predicted Price Changes')
plt.show()
