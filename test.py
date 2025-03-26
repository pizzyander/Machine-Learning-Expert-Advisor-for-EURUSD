import backtrader as bt
import requests
import json
import pandas as pd
import numpy as np
import asyncio
import joblib
from sklearn.preprocessing import MinMaxScaler
from metaapi_cloud_sdk import MetaApi
import os
from datetime import datetime, timedelta, timezone

# Load settings
with open('settings.json', 'r') as file:
    settings = json.load(file)

PREDICTION_URL = settings.get('prediction_url')
TOKEN = settings.get('metaapi_access_token')
ACCOUNT_ID = settings.get('metaapi_accountid')
SYMBOL = os.getenv('SYMBOL') or 'EURUSD'
DOMAIN = settings.get('domain') or 'agiliumtrade.agiliumtrade.ai'

class GRUBacktestStrategy(bt.Strategy):
    params = (('seq_length', 30),)

    def __init__(self):
        self.data_close = self.datas[0].close
        self.feature_window = []

    def next(self):
        current_data = {
            'close': self.data_close[0],
            'high': self.datas[0].high[0],
            'low': self.datas[0].low[0],
            'tickVolume': self.datas[0].volume[0]  
        }
        self.feature_window.append(current_data)

        if len(self.feature_window) < self.params.seq_length:
            return

        df_window = pd.DataFrame(self.feature_window)

        # ✅ Compute indicators & check for errors
        try:
            df_features = compute_indicators(df_window)
        except KeyError as e:
            print(f" Missing feature in compute_indicators: {e}")
            return

        # ✅ Scale features & check for errors
        try:
            input_data = scale_features(df_features)
        except KeyError as e:
            print(f" Feature mismatch error: {e}")
            return

        # ✅ Get prediction from API & check response
        prediction = self.get_prediction(input_data)
        print(f"Prediction: {prediction} at {self.datas[0].datetime.datetime(0)}")
        if prediction is not None:
            account_balance = self.broker.getvalue()
            stop_loss_pips = account_balance * 0.01 / 100_000  # Convert to forex pips
            take_profit = 60 * 0.0001  # 60 pips
            lot_size = 1.0  

            # ✅ BUY Trade
            if prediction > self.data_close[0]:
                if not self.position:
                    print(f" BUY Order placed at {self.data_close[0]}")
                    self.buy(size=lot_size, exectype=bt.Order.Market, 
                             trailamount=stop_loss_pips, tpprice=self.data_close[0] + take_profit)

            # ✅ SELL Trade
            elif prediction < self.data_close[0]:
                if self.position:
                    print(f" Closing BUY position at {self.data_close[0]}")
                    self.close()
                else:
                    print(f" SELL Order placed at {self.data_close[0]}")
                    self.sell(size=lot_size, exectype=bt.Order.Market, 
                              trailamount=stop_loss_pips, tpprice=self.data_close[0] - take_profit)

        self.feature_window.pop(0)

    def get_prediction(self, data):
        payload = {'features': data.tolist()}
        try:
            response = requests.post(PREDICTION_URL, json=payload)
            response_data = response.json()
            return response_data.get('prediction')
        except Exception as e:
            print(f" Error fetching prediction: {e}")
            return None

async def retrieve_historical_candles():
    api = MetaApi(TOKEN, {'domain': DOMAIN})
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)

    if account.state != 'DEPLOYED':
        await account.deploy()
    if account.connection_status != 'CONNECTED':
        await account.wait_connected()

    num_candles = 200
    start_time = datetime.now(timezone.utc)
    candles = []

    while len(candles) < num_candles:
        new_candles = await account.get_historical_candles(SYMBOL, '4h', start_time)
        if not new_candles:
            break
        candles.extend(new_candles)
        start_time = new_candles[0]['time'] - timedelta(hours=4)
        if len(candles) >= num_candles:
            break

    data = pd.DataFrame(candles)

    if not data.empty:
        print("Columns returned from API:", data.columns)  # Debugging line

        if 'tickVolume' not in data.columns and 'volume' in data.columns:
            data.rename(columns={'volume': 'tickVolume'}, inplace=True)
        
        # Fill missing values
        data['tickVolume'].fillna(0, inplace=True)
        
        data['time'] = pd.to_datetime(data['time'], unit='ms')
        data.set_index('time', inplace=True)
        data = data[['open', 'high', 'low', 'close', 'tickVolume']]

    print(f"Retrieved {len(data)} candles.")
    return data

def compute_indicators(data):
    if 'tickVolume' not in data.columns:
        raise KeyError("'tickVolume' is missing from the dataset")

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
    print("Columns in compute_indicators:", data.columns)

    if 'tickVolume' not in data.columns:
        raise KeyError("'tickVolume' is missing from the dataset")

    data['OBV'] = (np.sign(data['close'].diff()) * data['tickVolume']).cumsum()
    data['ADL'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']) * data['tickVolume']
    return data

def scale_features(data):
    if len(data) < 30:
        raise ValueError(f"Insufficient data: Expected at least 30 rows, got {len(data)}")
    scaler = "models/X_scaler.pkl"
    X_scaler = joblib.load(scaler)
    
    feature_columns = X_scaler.feature_names_in_
    data = data[feature_columns].fillna(0)
    
    scaled_data = X_scaler.transform(data[-30:])
    return scaled_data.reshape(1, 30, -1)

async def run_backtest():
    data = await retrieve_historical_candles()
    data = compute_indicators(data)
    data.fillna(0, inplace=True)

    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.addstrategy(GRUBacktestStrategy)

    # Add performance analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe_ratio")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    print("Starting Backtest...")
    start_balance = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    end_balance = cerebro.broker.getvalue()

    # Get analysis results
    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
    sharpe_ratio = strat.analyzers.sharpe_ratio.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    time_return = strat.analyzers.timereturn.get_analysis()

    # ✅ FIX: Ensure max drawdown is a float
    max_drawdown = drawdown.get('max', {}).get('drawdown', 0)  # Extract correct value
    max_drawdown_duration = drawdown.get('max', {}).get('len', 0)

    # Compute performance metrics
    total_trades = trade_analysis.get('total', {}).get('closed', 0)
    win_trades = trade_analysis.get('won', {}).get('total', 0)
    lose_trades = trade_analysis.get('lost', {}).get('total', 0)
    win_rate = (win_trades / total_trades * 100) if total_trades else 0
    total_return = ((end_balance - start_balance) / start_balance) * 100 if start_balance else 0
    benchmark_return = sum(time_return.values()) * 100  # Cumulative return

    # ✅ DEBUG: Check if the strategy received API predictions
    print("\n=== API Prediction Check ===")
    test_input = np.random.rand(30, 4)  # Simulate a feature input
    prediction = strat.get_prediction(test_input)
    print(f"Sample Prediction from API: {prediction}\n")

    # ✅ PRINT BACKTEST RESULTS
    print("\n=== Backtest Results ===")
    print(f"Start Date: {data.index[0] if not data.empty else 'N/A'}")
    print(f"End Date: {data.index[-1] if not data.empty else 'N/A'}")
    print(f"Period: {len(data)} candles")
    print(f"Start Balance: ${start_balance:.2f}")
    print(f"End Balance: ${end_balance:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Benchmark Return: {benchmark_return:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Max Drawdown Duration: {max_drawdown_duration} days")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio.get('sharperatio', 'N/A')}")
    print("========================\n")

# Run backtest asynchronously
asyncio.run(run_backtest())
