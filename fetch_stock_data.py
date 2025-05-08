import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker="AAPL", period="5y", interval="1d", save_path="data/data.csv"):
    
    # Downloads historical data
    data_frame = yf.download(ticker, period=period, interval=interval)

    # Drop any rows with missing values
    data_frame = data_frame.dropna()

    # Save the data locally
    data_frame.to_csv(save_path)

if __name__ == "__main__":
    fetch_stock_data()
