import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import csv
import os
import matplotlib.pyplot as plt
from metaapi_cloud_sdk import MetaApi

# Define the storage directory inside a mounted volume
MODELS_DIR = os.path.abspath("models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load credentials from a mounted settings.json file
settings_path = "settings.json"
with open(settings_path, 'r') as file:
    settings = json.load(file)

token = settings.get('metaapi_access_token')
account_id = settings.get('metaapi_accountid')
symbol = 'EURUSD'
domain = settings.get('domain') or 'agiliumtrade.agiliumtrade.ai'

# File path
csv_file_path = os.path.join(MODELS_DIR, "data.csv1")
import asyncio
import json
import pandas as pd
from datetime import datetime, timezone
from metaapi_cloud_sdk import MetaApi
import os

# Load credentials
settings_path = "settings.json"
with open(settings_path, 'r') as file:
    settings = json.load(file)

token = settings.get('metaapi_access_token')
account_id = settings.get('metaapi_accountid')
symbol = 'EURUSD'
domain = settings.get('domain', 'agiliumtrade.agiliumtrade.ai')


async def retrieve_historical_candles():
    """Fetch historical candles, save them as CSV, and return a Pandas DataFrame."""
    api = MetaApi(token, {'domain': domain})
    
    try:
        account = await api.metatrader_account_api.get_account(account_id)

        # Ensure the account is deployed and connected
        print('Deploying account...')
        if account.state != 'DEPLOYED':
            await account.deploy()
        else:
            print('Account already deployed')

        print('Waiting for API server to connect to broker...')
        if account.connection_status != 'CONNECTED':
            await account.wait_connected()

        # Retrieve last 10K 4H candles
        pages = 10
        data = []  # We'll use 'data' to store all candles
        start_time = None  # Start from the latest available data

        print(f'Downloading {pages}K latest candles for {symbol}')
        started_at = datetime.now().timestamp()

        for _ in range(pages):
            new_candles = await account.get_historical_candles(symbol, '4h', start_time)

            if new_candles:
                print(f'Downloaded {len(new_candles)} historical candles for {symbol}')
        
                # Append new data correctly to 'data'
                data.extend(new_candles)  

                # Fix: Move `start_time` to the **earliest** candle retrieved, not the latest
                start_time = new_candles[0]['time']  # Move to the oldest candle
                start_time -= timedelta(minutes=1)   # Move slightly earlier to avoid overlap

            else:
                print("No more candles available.")
                break  # Stop if no more data is returned

        if data:
            print(f'Total candles retrieved: {len(data)}')

            # Convert list of dictionaries to DataFrame
            df = pd.DataFrame(data)

            # Convert 'time' column to datetime
            df['time'] = pd.to_datetime(df['time'])

            return df  # Return DataFrame directly

    except Exception as err:
        print(api.format_error(err))
        return None
    
# Fetch data and plot
data = asyncio.run(retrieve_historical_candles())

def process_and_save_data(data, csv_filename="data.csv1", save_dir="models"):
    """
    Process the retrieved historical data:
    - Convert 'time' column to datetime
    - Sort by 'time' in ascending order
    - Keep only the first 3500 observations
    - Save the processed data as CSV
    """
    if data is None or data.empty:
        print("❌ No data to process.")
        return None

    # Convert 'time' column to datetime
    data['time'] = pd.to_datetime(data['time'])

    # Sort DataFrame by 'time' in ascending order
    data = data.sort_values(by='time', ascending=True)

    # Keep only the top 3500 observations
    data = data.iloc[3500:]

    # Define full CSV file path
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    csv_path = os.path.join(save_dir, csv_filename)

    # Save DataFrame to CSV
    data.to_csv(csv_path, index=False)

    return data  # Return the processed DataFrame

# Example usage after retrieving data
data = process_and_save_data(data)