import pandas as pd
import os
import matplotlib.pyplot as plt

# Load CSV file into a Pandas DataFrame
csv_file_path = "models/data.csv"  # Update the path if needed
data = pd.read_csv(csv_file_path)

# Convert 'time' column to datetime format
data['time'] = pd.to_datetime(data['time'])

# Sort DataFrame by 'time' in descending order
data = data.sort_values(by='time', ascending=False)

# Display the sorted DataFrame
print(data.head())
print(data.tail())

# Keep only the top 3500 observations
data = data.iloc[:3500]  # OR eurusd4hdata.head(3500)


# Define CSV file path
csv_path = os.path.join("models", "data.csv1")

# Save DataFrame to CSV
data.to_csv(csv_path, index=False)

print(f"Data saved to {csv_path}")

def plot_close_price(data):
   
    # Ensure 'time' is in datetime format
    data['time'] = pd.to_datetime(data['time'])

    # Plot close price over time
    plt.figure(figsize=(10, 6))
    plt.plot(data['time'], data['close'], label="Close Price", color='blue')
    plt.title("Close Price Over Time")
    plt.xlabel("Date and Time")
    plt.ylabel("Close Price")
    plt.grid(True)

    # Format the x-axis to show date and time clearly
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.legend()
    plt.show()

plot_close_price(data)