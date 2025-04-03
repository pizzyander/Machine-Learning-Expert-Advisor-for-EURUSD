import asyncio
import json
import os
import csv
import pandas as pd
from datetime import datetime, timezone
from metaapi_cloud_sdk import MetaApi

# Define storage directory
MODELS_DIR = os.path.abspath("models")
os.makedirs(MODELS_DIR, exist_ok=True)

# File path
csv_file_path = os.path.join(MODELS_DIR, "data.csv1")

# Load credentials from settings.json
settings_path = "settings.json"
with open(settings_path, 'r') as file:
    settings = json.load(file)

token = settings.get('metaapi_access_token')
account_id = settings.get('metaapi_accountid')
symbol = 'EURUSD'
domain = settings.get('domain', 'agiliumtrade.agiliumtrade.ai')


async def retrieve_historical_candles():
    """Fetches the most recent historical candles using MetaAPI"""
    
    api = MetaApi(token, {'domain': domain})
    account = await api.metatrader_account_api.get_account(account_id)

    if account.state != 'DEPLOYED':
        await account.deploy()
    if account.connection_status != 'CONNECTED':
        await account.wait_connected()

    num_candles = 10000  # Adjust as needed
    start_time = datetime.now(timezone.utc)  # Start fetching from the latest time

    # Fetch candles
    candles = await account.get_historical_candles(symbol, '4h', start_time)

    if not candles:
        print("No candles retrieved. Check your connection or symbol.")
        return pd.DataFrame()

    # Convert to DataFrame
    data = pd.DataFrame(candles)

    if not data.empty:
        data = data[['time', 'open', 'high', 'low', 'close', 'tickVolume']]
        data['time'] = pd.to_datetime(data['time'])  # Ensure proper datetime format
        data.sort_values('time', ascending=True, inplace=True)  # Ensure proper order

    print(f"Retrieved {len(data)} candles.") 
    return data
# Run the script
data = asyncio.run(retrieve_historical_candles())

if data is not None:
    print(data.tail())  # Display first few rows
    print(data.head())
# Create the 'models' directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Define CSV file path
csv_path = os.path.join("models", "data.csv1")

# Save DataFrame to CSV
data.to_csv(csv_path, index=False)

print(f"Data saved to {csv_path}")
