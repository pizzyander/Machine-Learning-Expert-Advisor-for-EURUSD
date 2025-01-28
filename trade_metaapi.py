import os
import asyncio
import json
from metaapi_cloud_sdk import MetaApi
from datetime import datetime, timedelta

# Load credentials from settings.json
with open('settings.json', 'r') as file:
    settings = json.load(file)

token = settings.get('metaapi_access_token')
login = settings.get('login')
password = settings.get('password')
server_name = settings.get('server')


async def meta_api_synchronization():
    api = MetaApi(token)
    try:
        # Add test MetaTrader account
        accounts = await api.metatrader_account_api.get_accounts_with_infinite_scroll_pagination()
        account = None
        for item in accounts:
            if item.login == str(login) and item.type.startswith('cloud'):
                account = item
                break
        if not account:
            print('Adding MT5 account to MetaApi')
            account = await api.metatrader_account_api.create_account(
                {
                    'name': 'Test account',
                    'type': 'cloud',
                    'login': str(login),
                    'password': password,
                    'server': server_name,
                    'platform': 'mt5',
                    'application': 'MetaApi',
                    'magic': 1000,
                }
            )
        else:
            print('MT5 account already added to MetaApi')

        # Wait until account is deployed and connected to broker
        print('Deploying account')
        await account.deploy()
        print('Waiting for API server to connect to broker (may take a couple of minutes)')
        await account.wait_connected()

        # Connect to MetaApi API
        connection = account.get_rpc_connection()
        await connection.connect()

        # Wait until terminal state is synchronized to the local state
        await connection.wait_synchronized()

        # Submit a market buy order (instant execution)
        print('Submitting market buy order')
        try:
            result = await connection.create_market_buy_order(
                'GBPUSD',  # Symbol
                0.01,       # Volume (lot size)
                1.0,       # Stop loss (optional, set to 0 to disable)
                2.0,       # Take profit (optional, set to 0 to disable)
                {
                    'comment': 'Buy',  # Shortened comment
                    'clientId': 'TE_GBPUSD_123'  # Shortened clientId
                }
            )
            print('Trade successful, result code is ' + result['stringCode'])
        except Exception as err:
            print('Trade failed with error:')
            print(api.format_error(err))

        # Finally, undeploy the account after the test
        print('Undeploying MT5 account so that it does not consume any unwanted resources')
        await connection.close()
        await account.undeploy()

    except Exception as err:
        # Process errors
        if hasattr(err, 'details'):
            if err.details == 'E_SRV_NOT_FOUND':
                print(err)
            elif err.details == 'E_AUTH':
                print(err)
            elif err.details == 'E_SERVER_TIMEZONE':
                print(err)
        print(api.format_error(err))
    exit()


asyncio.run(meta_api_synchronization())