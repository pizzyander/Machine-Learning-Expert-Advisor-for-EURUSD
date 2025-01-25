import MetaTrader5 as mt5
import requests
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    filename="trading_bot.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Global Variables
SYMBOL = "EURUSD"
USERNAME = 9293182
PASSWORD = "Ge@mK3Xb"
SERVER = "GTCGlobalTrade-Server"
MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
FASTAPI_URL = "http://127.0.0.1:8000/predict"  # Update with actual FastAPI URL

def start_mt5(username, password, server, path):
    """Initialize and log in to MT5."""
    logging.info("Initializing MT5...")
    if not mt5.initialize(path=path):
        logging.error(f"MT5 initialization failed: {mt5.last_error()}")
        return False

    if not mt5.login(username, password, server):
        logging.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    logging.info("MT5 initialized and logged in successfully.")
    return True

def initialize_symbols(symbols):
    """Enable required trading symbols."""
    logging.info("Initializing symbols...")
    for symbol in symbols:
        if mt5.symbol_select(symbol, True):
            logging.info(f"Symbol {symbol} enabled.")
        else:
            logging.error(f"Failed to enable symbol {symbol}: {mt5.last_error()}")
            return False
    return True

def get_last_90_candles(symbol):
    """Retrieve the last 90 H1 candlestick data."""
    logging.info(f"Retrieving last 90 H1 candles for symbol: {symbol}")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H6, 0, 360)

    if rates is None or len(rates) == 0:
        logging.error("Failed to retrieve candlestick data.")
        return None

    logging.info("Candlestick data retrieved successfully.")
    # Return candle data with clear naming
    candle_data = [
        {"open": rate[1], "high": rate[2], "low": rate[3], "close": rate[4]}
        for rate in rates
    ]
    return candle_data


def get_prediction(candle_data):
    """Send candlestick data to FastAPI for prediction."""
    logging.info("Sending data to FastAPI for prediction...")
    try:
        # Prepare features with the required columns: open, high, low
        features = [{"open": candle["open"], "high": candle["high"], "low": candle["low"]} for candle in candle_data]
        
        # Log the prepared features
        logging.debug(f"Prepared features: {features}")
        
        # Send data to FastAPI
        response = requests.post(FASTAPI_URL, json={"features": features})
        
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            logging.info(f"Prediction received: {prediction}")
            return prediction
        else:
            logging.error(f"Failed to get prediction. Status Code: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        logging.error(f"Error during FastAPI request: {e}")
        return None

def place_trade_with_programmatic_tp(symbol, order_type):
    """Place a trade and manage programmatic take-profit levels."""
    lot_size = 0.3
    price = mt5.symbol_info_tick(symbol).ask if order_type == "buy" else mt5.symbol_info_tick(symbol).bid
    point = mt5.symbol_info(symbol).point
    tp_levels = [
        price + (120 * point) if order_type == "buy" else price - (120 * point),
        price + (240 * point) if order_type == "buy" else price - (240 * point),
        price + (360 * point) if order_type == "buy" else price - (360 * point),
    ]
    tp_lot_sizes = [0.1, 0.1, 0.1]

    logging.info(f"Placing {order_type.upper()} order for {symbol}.")
    try:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 10,
            "magic": 123456,
            "comment": "Programmatic TP",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order placement failed: {result.retcode}")
            return

        logging.info(f"Order placed successfully. Ticket: {result.order}")
        ticket = result.order

        for i, tp in enumerate(tp_levels):
            while True:
                current_price = (
                    mt5.symbol_info_tick(symbol).bid if order_type == "buy" else mt5.symbol_info_tick(symbol).ask
                )
                if (order_type == "buy" and current_price >= tp) or (
                    order_type == "sell" and current_price <= tp
                ):
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": tp_lot_sizes[i],
                        "type": mt5.ORDER_TYPE_SELL if order_type == "buy" else mt5.ORDER_TYPE_BUY,
                        "position": ticket,
                        "price": current_price,
                        "deviation": 10,
                        "magic": 123456,
                        "comment": f"TP Level {i+1} closure",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    close_result = mt5.order_send(close_request)

                    if close_result.retcode != mt5.TRADE_RETCODE_DONE:
                        logging.error(f"Failed to close at TP Level {i+1}: {close_result.retcode}")
                    else:
                        logging.info(f"Successfully closed {tp_lot_sizes[i]} lots at TP Level {i+1}.")
                    break

                time.sleep(1)
    except Exception as e:
        logging.error(f"Error during trade placement or TP management: {e}")

def main():
    logging.info("Starting main trading bot function...")
    if not start_mt5(USERNAME, PASSWORD, SERVER, MT5_PATH):
        return

    if not initialize_symbols([SYMBOL]):
        return

    candle_data = get_last_90_candles(SYMBOL)
    if candle_data is None:
        return

    prediction = get_prediction(candle_data)
    if prediction is None:
        return

    last_close = candle_data[-1]["close"]
    if prediction > last_close:
        logging.info("Prediction indicates a BUY signal.")
        place_trade_with_programmatic_tp(SYMBOL, "buy")
    else:
        logging.info("Prediction indicates a SELL signal.")
        place_trade_with_programmatic_tp(SYMBOL, "sell")

    logging.info("Trading bot cycle complete.")

if __name__ == "__main__":
    main()
