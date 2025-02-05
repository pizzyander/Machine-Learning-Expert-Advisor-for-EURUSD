import os
import asyncio
from metaapi_cloud_sdk import MetaApi
from datetime import datetime, timedelta, timezone
import json
import pandas as pd  # Import pandas

# Load credentials from settings.json
with open('settings.json', 'r') as file:
    settings = json.load(file)

# Extract settings
token = settings.get('metaapi_access_token')
account_id = settings.get('metaapi_accountid')
symbol = os.getenv('SYMBOL') or 'EURUSD'
domain = settings.get('domain') or 'agiliumtrade.agiliumtrade.ai'  # Default to MetaApi domain

async def retrieve_historical_candles():
    api = MetaApi(token, {'domain': domain})

    try:
        # Retrieve account details
        account = await api.metatrader_account_api.get_account(account_id)

        # Ensure account is deployed
        print('Deploying account...')
        if account.state != 'DEPLOYED':
            await account.deploy()
        else:
            print('Account already deployed')

        # Wait for connection to the broker
        print('Waiting for API server to connect to broker...')
        if account.connection_status != 'CONNECTED':
            await account.wait_connected()

        # Retrieve last 12,000 six-hour candles (~3.5 years of data)
        num_candles = 12000
        print(f'Downloading {num_candles} latest H6 candles for {symbol}')
        start_time = datetime.now(timezone.utc)  # Proper timezone-aware UTC datetime
        # Start from current time
        candles = []

        started_at = datetime.now().timestamp()

        while len(candles) < num_candles:
            # Retrieve historical candles
            new_candles = await account.get_historical_candles(symbol, '4h', start_time)

            if not new_candles:
                print(f'No more historical data available. Stopping.')
                break

            candles.extend(new_candles)
            print(f'Downloaded {len(new_candles)} candles. Total so far: {len(candles)}')

            # Update start_time for the next batch
            start_time = new_candles[0]['time'] - timedelta(hours=4)

            # Stop if we have enough data
            if len(candles) >= num_candles:
                candles = candles[:num_candles]  # Trim excess candles
                break

        if candles:
            print(f'First candle: {candles[0]}')
            print(f'Total retrieved: {len(candles)}')

        # Convert candles to DataFrame
        data = pd.DataFrame(candles)

        print(f'Execution time: {(datetime.now().timestamp() - started_at) * 1000:.2f} ms')

    except Exception as err:
        print(api.format_error(err))

# Run the function
asyncio.run(retrieve_historical_candles())