import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from risk_free_rate import get_india_10y_risk_free_rate

# Load data
data = pd.read_csv('../data/stock_prices.csv', index_col=0, parse_dates=True)
daily_returns = data.pct_change().dropna()
cov_matrix = daily_returns.cov()

def portfolio_stats(weights, returns, cov_matrix, risk_free_rate):
    """Calculate portfolio return, risk, and Sharpe ratio"""
    portfolio_return = np.sum(weights * returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe

def negative_sharpe(weights, returns, cov_matrix, risk_free_rate):
    """Minimize negative Sharpe to maximize Sharpe"""
    return -portfolio_stats(weights, returns, cov_matrix, risk_free_rate)[2]

# Optimization setup
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0, 1) for _ in range(len(daily_returns.columns)))
initial_guess = np.array([1/len(daily_returns.columns)] * len(daily_returns.columns))
risk_free_rate, rf_source = get_india_10y_risk_free_rate()

# Optimize for max Sharpe ratio
print("\nOptimizing portfolio (this takes a few seconds)...")
result = minimize(negative_sharpe, initial_guess,
                 args=(daily_returns, cov_matrix, risk_free_rate),
                 method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x

# Get stats
opt_return, opt_risk, opt_sharpe = portfolio_stats(optimal_weights, daily_returns, cov_matrix, risk_free_rate)
equal_weights = np.array([1/len(daily_returns.columns)] * len(daily_returns.columns))
eq_return, eq_risk, eq_sharpe = portfolio_stats(equal_weights, daily_returns, cov_matrix, risk_free_rate)

print("\n" + "="*80)
print("PORTFOLIO COMPARISON")
print("="*80)
print(f"Risk-free rate used: {risk_free_rate*100:.2f}% ({rf_source})")

print("\nEQUAL WEIGHT (Naive - 4.76% each)")
print(f"   Return: {eq_return*100:.2f}%  |  Risk: {eq_risk*100:.2f}%  |  Sharpe: {eq_sharpe:.2f}")

print("\nOPTIMAL WEIGHT (Data-Driven)")
print(f"   Return: {opt_return*100:.2f}%  |  Risk: {opt_risk*100:.2f}%  |  Sharpe: {opt_sharpe:.2f}")

print("\n" + "="*80)
print("OPTIMAL ALLOCATION")
print("="*80)

weights_df = pd.DataFrame({
    'Stock': daily_returns.columns,
    'Weight (%)': optimal_weights * 100
}).sort_values('Weight (%)', ascending=False)

# Show only stocks with >0.5% allocation
print("\nStocks with meaningful allocation (>0.5%):\n")
for idx, row in weights_df[weights_df['Weight (%)'] > 0.5].iterrows():
    print(f"  {row['Stock']:12s}  {row['Weight (%)']:6.2f}%")

print("\n" + "-"*80)
print(f"Total allocation: {weights_df['Weight (%)'].sum():.2f}%")

# Save
weights_df.to_csv('../data/optimal_weights.csv', index=False)
print("\nSaved optimal_weights.csv")

# Portfolio improvement
improvement = ((opt_sharpe - eq_sharpe) / eq_sharpe) * 100
print("\n" + "="*80)
print(f"PORTFOLIO IMPROVEMENT: Sharpe ratio improved by {improvement:.1f}%")
print("="*80)
