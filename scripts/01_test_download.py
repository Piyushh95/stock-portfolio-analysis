import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define stocks
stocks = {
    'US': ['AAPL', 'MSFT', 'TSLA'],
    'India': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
}

all_stocks = stocks['US'] + stocks['India']

# Fetch data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)

data = yf.download(all_stocks, start=start_date, end=end_date)

# FIX: Access 'Adj Close' correctly for multiple stocks
data = data['Adj Close']

# Check if data is empty or has issues
print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"\nFirst few rows:")
print(data.head())

# Save for later
data.to_csv('../data/stock_prices.csv')
