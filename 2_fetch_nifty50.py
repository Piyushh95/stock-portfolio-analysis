import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from data_sources.constituents import get_nifty_constituents

# Fetch NIFTY universe dynamically
nifty_50_stocks, source = get_nifty_constituents()

# Add US tech for diversity
us_stocks = ['AAPL', 'MSFT', 'GOOGL']
include_us = os.getenv("INCLUDE_US_TECH", "1") == "1"

all_stocks = nifty_50_stocks + us_stocks if include_us else nifty_50_stocks

end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)

print(f"Constituent source: {source}")
print(f"Downloading {len(all_stocks)} stocks... (2 years of data)\n")
data = yf.download(all_stocks, start=start_date, end=end_date)

# Fix structure
data = data['Close']
data = data.dropna(axis=1, how='all')
data = data.ffill()

print(f"\nData shape: {data.shape}")
print(f"Stocks ({len(data.columns)}): {list(data.columns)}")

# Save universe metadata for traceability
os.makedirs("analysis", exist_ok=True)
metadata = pd.DataFrame(
    {
        "symbol": all_stocks,
        "constituent_source": source,
        "include_us_tech": include_us,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
)
metadata.to_csv("analysis/universe_metadata.csv", index=False)

# Save
data.to_csv('data/stock_prices.csv')
print("\nSaved to data/stock_prices.csv")