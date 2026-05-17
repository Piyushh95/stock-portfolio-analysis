import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from risk_free_rate import get_india_10y_risk_free_rate

# Load data
data = pd.read_csv('../data/stock_prices.csv', index_col=0, parse_dates=True)

# === RETURNS ===
daily_returns = data.pct_change().dropna()
annual_returns = daily_returns.mean() * 252

# === RISK ===
annual_volatility = daily_returns.std() * np.sqrt(252)

# === CORRELATION ===
correlation_matrix = daily_returns.corr()

# === PERFORMANCE ===
total_return = (data.iloc[-1] / data.iloc[0] - 1) * 100

# === SHARPE RATIO ===
risk_free_rate, rf_source = get_india_10y_risk_free_rate()
sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility

# === Create Summary ===
summary = pd.DataFrame({
    'Stock': daily_returns.columns,
    'Total Return (%)': total_return.values,
    'Annual Return (%)': annual_returns.values * 100,
    'Annual Volatility (%)': annual_volatility.values * 100,
    'Sharpe Ratio': sharpe_ratio.values
}).sort_values('Sharpe Ratio', ascending=False).round(2)

print("\n" + "="*80)
print("STOCK ANALYSIS SUMMARY (21 Stocks - Nifty 50 + US Tech)")
print("="*80)
print(summary.to_string(index=False))

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"Risk-free rate used: {risk_free_rate*100:.2f}% ({rf_source})")
print(f"\nBest Performer: {summary.iloc[0]['Stock']} ({summary.iloc[0]['Total Return (%)']:.2f}%)")
print(f"Worst Performer: {summary.iloc[-1]['Stock']} ({summary.iloc[-1]['Total Return (%)']:.2f}%)")
print(f"\nBest Risk-Adjusted (Sharpe): {summary.iloc[0]['Stock']} ({summary.iloc[0]['Sharpe Ratio']:.2f})")
print(f"Worst Risk-Adjusted (Sharpe): {summary.iloc[-1]['Stock']} ({summary.iloc[-1]['Sharpe Ratio']:.2f})")

# Save for later
summary.to_csv('../analysis/stock_summary.csv', index=False)
correlation_matrix.to_csv('../analysis/correlation_matrix.csv')

print("\nSaved stock_summary.csv and correlation_matrix.csv")
