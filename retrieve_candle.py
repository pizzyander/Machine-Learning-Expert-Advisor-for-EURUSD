import os
import asyncio
from metaapi_cloud_sdk import MetaApi
from datetime import datetime
import json

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
        print('Deploying account')
        if account.state != 'DEPLOYED':
            await account.deploy()
        else:
            print('Account already deployed')

        # Wait for connection to the broker
        print('Waiting for API server to connect to broker (may take a couple of minutes)')
        if account.connection_status != 'CONNECTED':
            await account.wait_connected()

        # Retrieve last 10K 1-minute candles
        pages = 10
        print(f'Downloading {pages}K latest candles for {symbol}')
        started_at = datetime.now().timestamp()
        start_time = None
        candles = None

        for i in range(pages):
            # Retrieve historical candles
            new_candles = await account.get_historical_candles(symbol, '1m', start_time)
            print(f'Downloaded {len(new_candles) if new_candles else 0} historical candles for {symbol}')

            if new_candles and len(new_candles):
                candles = new_candles

            if candles and len(candles):
                # Adjust start time for the next batch
                start_time = candles[0]['time']
                print(f'First candle time is {start_time}')

        if candles:
            print(f'First candle is', candles[0])
        print(f'Took {(datetime.now().timestamp() - started_at) * 1000:.2f} ms')

    except Exception as err:
        print(api.format_error(err))
    exit()

# Run the function
asyncio.run(retrieve_historical_candles())
