import pandas as pd
import os

def merge_data():
    # Define the storage directory inside a mounted volume
    MODELS_DIR = os.path.abspath("models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # File path
    csv1 = os.path.join(MODELS_DIR, "data.csv1")
    csv2 = os.path.join(MODELS_DIR, "data.csv2")

    # Load both datasets
    df2 = pd.read_csv(csv2)
    df1 = pd.read_csv(csv1)

    # Convert to datetime
    df1['time'] = pd.to_datetime(df1['time'])
    df2['time'] = pd.to_datetime(df2['time'])

    # Remove timezone info to make both timezone-naive
    df1['time'] = df1['time'].dt.tz_localize(None)
    df2['time'] = df2['time'].dt.tz_localize(None)

    # Dynamically get the latest timestamp from csv2
    latest_timestamp = df2['time'].max()
    print(f"Latest timestamp in csv2: {latest_timestamp}")

    # Filter recent data from that timestamp onward
    df1_filtered = df1[df1['time'] > latest_timestamp]

    # Combine the two datasets
    merged_df = pd.concat([df2, df1_filtered], ignore_index=True)

    #Drop duplicates if needed
    merged_df = merged_df.drop_duplicates(subset='time')

    # Save the merged dataset to the models directory
    output_path = os.path.join(MODELS_DIR, "merged_data.csv")
    merged_df.to_csv(output_path, index=False)

    # Display head and tail
    print(f"Merged data saved to {output_path}")
    print(merged_df.head())
    print(merged_df.tail())
    return merged_df

merge_data()