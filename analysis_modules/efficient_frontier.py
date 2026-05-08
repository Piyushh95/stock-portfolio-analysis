import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate):
    expected_return = np.dot(weights, mean_returns) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (expected_return - risk_free_rate) / volatility if volatility > 0 else np.nan
    return expected_return, volatility, sharpe


def _max_sharpe_weights(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial = np.array([1 / num_assets] * num_assets)

    def objective(weights):
        return -_portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[2]

    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        return initial
    return result.x


def generate_efficient_frontier(
    prices_df,
    risk_free_rate,
    n_portfolios=2000,
    seed=42,
    output_path="analysis/efficient_frontier.csv",
):
    returns = prices_df.pct_change().dropna()
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    tickers = list(returns.columns)

    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_portfolios):
        w = rng.random(len(tickers))
        w = w / w.sum()
        exp_ret, vol, sharpe = _portfolio_stats(w, mean_returns, cov_matrix, risk_free_rate)
        row = {
            "portfolio_type": "random",
            "expected_return": exp_ret,
            "volatility": vol,
            "sharpe": sharpe,
        }
        for t, val in zip(tickers, w):
            row[f"w_{t}"] = val
        rows.append(row)

    # Include deterministic max-sharpe portfolio as reference.
    max_w = _max_sharpe_weights(mean_returns, cov_matrix, risk_free_rate)
    max_ret, max_vol, max_sharpe = _portfolio_stats(max_w, mean_returns, cov_matrix, risk_free_rate)
    max_row = {
        "portfolio_type": "max_sharpe",
        "expected_return": max_ret,
        "volatility": max_vol,
        "sharpe": max_sharpe,
    }
    for t, val in zip(tickers, max_w):
        max_row[f"w_{t}"] = val
    rows.append(max_row)

    frontier_df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frontier_df.to_csv(output_path, index=False)
    return frontier_df
