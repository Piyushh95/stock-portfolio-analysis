import os
from datetime import datetime

import pandas as pd
import requests


NSE_NIFTY50_URL = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
NSE_HOME_URL = "https://www.nseindia.com/"
CACHE_PATH = "data/constituents_cache.csv"

DEFAULT_NIFTY50_FALLBACK = [
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "TCS.NS",
    "INFY.NS",
    "WIPRO.NS",
    "LTTS.NS",
    "RELIANCE.NS",
    "POWERGRID.NS",
    "MARUTI.NS",
    "TATAMOTORS.NS",
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "NESTLEIND.NS",
    "ULTRACEMCO.NS",
    "SHREECEM.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
]


def _nse_session():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/",
        }
    )
    # Prime cookies for API access
    session.get(NSE_HOME_URL, timeout=10)
    return session


def fetch_nse_nifty50():
    session = _nse_session()
    response = session.get(NSE_NIFTY50_URL, timeout=15)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if not data:
        raise ValueError("NSE response did not include constituents.")

    symbols = []
    for row in data:
        symbol = row.get("symbol")
        if symbol:
            symbols.append(f"{symbol}.NS")

    symbols = sorted(set(symbols))
    if not symbols:
        raise ValueError("No valid symbols parsed from NSE payload.")
    return symbols


def read_cached_constituents(cache_path=CACHE_PATH):
    if not os.path.exists(cache_path):
        raise FileNotFoundError("Constituents cache file not found.")
    cache_df = pd.read_csv(cache_path)
    if "symbol" not in cache_df.columns:
        raise ValueError("Cache missing expected 'symbol' column.")
    symbols = cache_df["symbol"].dropna().astype(str).tolist()
    if not symbols:
        raise ValueError("Constituents cache is empty.")
    return symbols


def write_constituents_cache(symbols, cache_path=CACHE_PATH, source="nse"):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pd.DataFrame(
        {
            "symbol": symbols,
            "source": source,
            "fetched_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }
    ).to_csv(cache_path, index=False)


def get_nifty_constituents():
    """
    Return (symbols, source) with robust fallback behavior.
    Source priority: NSE API -> local cache -> static fallback.
    """
    try:
        symbols = fetch_nse_nifty50()
        write_constituents_cache(symbols, source="nse_api")
        return symbols, "nse_api"
    except Exception:
        try:
            symbols = read_cached_constituents()
            return symbols, "cache_file"
        except Exception:
            return DEFAULT_NIFTY50_FALLBACK, "static_fallback"
