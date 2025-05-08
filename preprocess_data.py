import pandas as pd
import numpy as np

def preprocess_data(csv_path="data/data.csv", sequence_length=30):

    # Load the stored data from the CSV file
    data_frame = pd.read_csv(csv_path)

    # Extract only the 'Close' prices
    close_prices = data_frame['Close'].values
    
    x = [] # x will be the input sequences (Closing Prices for the last 30 days)
    
    y = [] # y will be the target values (Closing Price on the next day after the 30 days)

    # Create sliding windows of the closing prices
    # For example, if sequence_length is 30, then for the first 30 days, the target will be the 31st day
    # This means that for each sequence of 30 days, we will predict the next day's closing price
    for i in range(len(close_prices) - sequence_length):
        x.append(close_prices[i:i + sequence_length]) # Append the sequence of past 30 days of closing prices
        y.append(close_prices[i + sequence_length]) # Append the next day's closing price

    x = np.array(x) # Convert x to a numpy array
    y = np.array(y) # Convert y to a numpy array

    # Verifying data
    print(f"Data shapes -> X: {x.shape}, y: {y.shape}")

    return x, y

if __name__ == "__main__":
    preprocess_data()





