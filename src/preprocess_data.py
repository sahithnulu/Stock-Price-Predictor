import pandas as pd
import numpy as np

def preprocess_data(csv_path="./data/data.csv", sequence_length=30):
    # Load the stored data from the CSV file
    data_frame = pd.read_csv(csv_path)

    # Remove any rows that cannot be converted to numeric (e.g., headers like 'AAPL', 'Ticker')
    data_frame = data_frame[pd.to_numeric(data_frame['Close'], errors='coerce').notna()]
    
    # Convert 'Close' column to float type
    data_frame['Close'] = data_frame['Close'].astype(float)

    # Extract only the cleaned 'Close' prices
    close_prices = data_frame['Close'].values

    x, y = [], []

    # Create sliding windows
    for i in range(len(close_prices) - sequence_length):
        x.append(close_prices[i:i + sequence_length])
        y.append(close_prices[i + sequence_length])

    x = np.array(x)
    y = np.array(y)

    print(f"Data shapes -> X: {x.shape}, y: {y.shape}")
    return x, y

if __name__ == "__main__":
    preprocess_data()




