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
        print('Waiting for SDK to synchronize to terminal state (may take some time depending on your history size)')
        await connection.wait_synchronized()

        # Invoke RPC API (replace ticket numbers with actual ticket numbers that exist in your MT account)
        print('Testing MetaAPI RPC API')
        print('Account information:', await connection.get_account_information())
        print('Positions:', await connection.get_positions())
        # Uncomment and replace with a valid ticket number
        # print(await connection.get_position('1234567'))
        print('Open orders:', await connection.get_orders())
        # Uncomment and replace with a valid order ID
        # print(await connection.get_order('1234567'))
        print('History orders by ticket:', await connection.get_history_orders_by_ticket('1234567'))
        print('History orders by position:', await connection.get_history_orders_by_position('1234567'))
        print(
            'History orders (~last 3 months):',
            await connection.get_history_orders_by_time_range(
                datetime.utcnow() - timedelta(days=90), datetime.utcnow()
            ),
        )
        print('History deals by ticket:', await connection.get_deals_by_ticket('1234567'))
        print('History deals by position:', await connection.get_deals_by_position('1234567'))
        print(
            'History deals (~last 3 months):',
            await connection.get_deals_by_time_range(datetime.utcnow() - timedelta(days=90), datetime.utcnow()),
        )

        print('Server time:', await connection.get_server_time())

        # Calculate margin required for trade
        print(
            'Margin required for trade:',
            await connection.calculate_margin(
                {'symbol': 'GBPUSD', 'type': 'ORDER_TYPE_BUY', 'volume': 0.1, 'openPrice': 1.1}
            ),
        )

        # Submit a pending order
        print('Submitting pending order')
        try:
            result = await connection.create_limit_buy_order(
                'GBPUSD', 0.07, 1.0, 0.9, 2.0, {'comment': 'comm', 'clientId': 'TE_GBPUSD_7hyINWqAlE'}
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
