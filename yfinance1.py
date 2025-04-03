import yfinance as yf
import os

# Fetch EUR/USD H4 data
eurusd4hdata = yf.download('EURUSD=X', start='2023-05-01', end='2025-04-03', interval='4h')

# Reset index to make datetime a column
eurusd4hdata.reset_index(inplace=True)

# Define CSV file path
csv_path = os.path.join("models", "eurusd_h4_data.csv")

# Ensure the directory exists
os.makedirs("models", exist_ok=True)

# Save DataFrame to CSV
eurusd4hdata.to_csv(csv_path, index=False)

print(f"Data saved to {csv_path}")
print(eurusd4hdata.tail())
