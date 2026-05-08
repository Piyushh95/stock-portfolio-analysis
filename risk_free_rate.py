from io import StringIO

import pandas as pd
import requests


DEFAULT_RISK_FREE_RATE = 0.05
FRED_SERIES_ID = "INDIRLTLT01STM"  # India 10Y government bond yield (%)
FRED_CSV_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_SERIES_ID}"


def get_india_10y_risk_free_rate(default_rate=DEFAULT_RISK_FREE_RATE):
    """
    Return annualized risk-free rate as a decimal.

    Tries to fetch India 10Y yield from FRED and falls back to default.
    """
    try:
        response = requests.get(FRED_CSV_URL, timeout=10)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        df = df[df[FRED_SERIES_ID] != "."].copy()
        if df.empty:
            raise ValueError("No valid FRED values returned.")

        latest_pct = float(df[FRED_SERIES_ID].iloc[-1])
        latest_rate = latest_pct / 100.0

        if not (0.0 < latest_rate < 0.30):
            raise ValueError(f"Fetched rate out of expected range: {latest_rate}")

        return latest_rate, f"FRED:{FRED_SERIES_ID}"
    except Exception:
        return default_rate, "fallback:default_5pct"
