import pandas as pd
import numpy as np

# Load the data
csv_file_path = r"C:\Users\hp\Downloads\EURUSD_H4.csv"
data = pd.read_csv(csv_file_path)

# Convert 'time' column to datetime
data['time'] = pd.to_datetime(data['time'])

# Sort DataFrame by 'time' in descending order
data = data.sort_values(by='time', ascending=True)

# Get last close price (for Open in new row)
last_close_price = data.iloc[0]['close']
print(last_close_price)

# Compute mean for 'high', 'low', 'tickvolume' from last 2 observations
mean_high = data.iloc[:2]['high'].mean()
mean_low = data.iloc[:2]['low'].mean()
mean_tickvolume = data.iloc[:2]['tickVolume'].mean()

# Create a new empty row (NaN values)
new_row = {col: np.nan for col in data.columns}

# Fill in required values
new_row['time'] = data.iloc[-1]['time'] + pd.Timedelta(hours=4)  # Assuming 4H timeframe
new_row['open'] = last_close_price  # Last close → Open
new_row['high'] = mean_high
new_row['low'] = mean_low
new_row['tickVolume'] = mean_tickvolume

# Append the new row at the bottom
data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

# Display the updated DataFrame
print(data.tail())  # Check last few rows

print(data.columns)

