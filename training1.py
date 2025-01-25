import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from datetime import datetime
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
import logging
from datetime import datetime
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


now = datetime.now()
today_date = str(now.date())


# Fetch historical data from MetaTrader 5
def fetch_mt5_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H6, start_date="2019-01-01", end_date = today_date):
    if not mt5.initialize():
        raise RuntimeError("Failed to initialize MetaTrader5, error code:", mt5.last_error())
    print("MetaTrader5 initialized successfully.")
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    mt5.shutdown()

    if rates is None:
        raise ValueError("No data fetched. Please check symbol and timeframe.")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    print("Data fetched successfully.")
    return df


# Clean and fill missing values in the dataset
def clean_data(df):
    df = df.resample("1H").mean()
    df.interpolate(method="linear", inplace=True)
    df = df[['open', 'high', 'low', 'close']]  # Fix column selection
    print("Data cleaned and missing values extrapolated.")
    return df

# Feature Engineering function
def add_features(df):
    df['SMA_20'] = df['open'].rolling(window=20).mean()
    df['EMA_50'] = df['open'].ewm(span=50, adjust=False).mean()
    delta = df['open'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['open'].ewm(span=12, adjust=False).mean() - df['open'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['volatility_20'] = df['open'].pct_change().rolling(window=20).std()
    return df

def create_new_df(df):
    df = add_features(df)
    df = df.dropna()
    return df
    
# Prepare data for training
def prepare_data(df):
    time_steps = 90
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = X_scaler.fit_transform(df.drop(columns='close'))
    y_scaled = y_scaler.fit_transform(df['close'].values.reshape(-1, 1))

    X, y = [], []
    X_sequenced = X_scaled
    y_sequenced = y_scaled

    for i in range(time_steps, len(df)):
        X.append(X_sequenced[i-time_steps:i])
        y.append(y_sequenced[i])

    return np.array(X), np.array(y), X_scaler, y_scaler

# Split data into training, validation, and test sets
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print("Data split into train, validation, and test sets.")
    return X_train, X_val, X_test, y_train, y_val, y_test

# Train GRU model
def train_model(X_train, y_train, X_val, y_val, save_path="gru_model.h5"):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        GRU(50),
        Dropout(0.2),
        Dense(1, activation='linear')  # Use linear for regression
    ])
    early_stop = EarlyStopping(monitor="val_loss", patience=4)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])  # Use MSE for regression
    
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])
    print("Training complete. History:", history.history)
    model.save(save_path)
    print(f"Model saved at {save_path}.")
    return model, history

# Evaluate the model
def evaluate_model(model, X_test, y_test, y_scaler):
    y_pred = model.predict(X_test)
    y_pred_inversed = y_scaler.inverse_transform(y_pred)
    y_test_inversed = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    print("Predictions vs. Actual:")
    print(f"Predicted: {y_pred_inversed[:5].flatten()}")
    print(f"Actual: {y_test_inversed[:5].flatten()}")
    return y_pred_inversed, y_test_inversed

# Save scalers and model
def save_scalers_and_model(X_scaler, y_scaler, model):
    dump(X_scaler, "X_train_scaled.joblib")
    dump(y_scaler, "y_train_scaled.joblib")
    model.save("gru_model.h5")
    return True

def main():
    # Parameters
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_H1
    start_date = "2019-01-01"
    end_date = today_date
    model_save_path = "gru_model.h5"

    try:
        # Step 1: Fetch historical data
        print("Fetching data...")
        raw_data = fetch_mt5_data(symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date)
        
        # Step 2: Clean and preprocess data
        print("Cleaning data...")
        cleaned_data = clean_data(raw_data)

        # Step 3: Add features
        print("Adding features...")
        featured_data = create_new_df(cleaned_data)

        # Step 4: Prepare data for training
        print("Preparing data for training...")
        X, y, X_scaler, y_scaler = prepare_data(featured_data)

        # Step 5: Split data
        print("Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # Step 6: Train the model
        print("Training the model...")
        model, history = train_model(X_train, y_train, X_val, y_val, save_path=model_save_path)

        # Step 7: Evaluate the model
        print("Evaluating the model...")
        y_pred, y_test_actual = evaluate_model(model, X_test, y_test, y_scaler)

        # Step 8: Save scalers and model
        print("Saving scalers and model...")
        save_scalers_and_model(X_scaler, y_scaler, model)

        # Step 9: Output results
        print("Model evaluation results:")
        print(f"Predicted Values: {y_pred[:5].flatten()}")
        print(f"Actual Values: {y_test_actual[:5].flatten()}")
        print("Pipeline executed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"Pipeline error: {e}")

if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs
    main()
