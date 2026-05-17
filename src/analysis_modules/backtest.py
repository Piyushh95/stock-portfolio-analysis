import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _max_sharpe_weights(train_returns, risk_free_rate):
    mean_returns = train_returns.mean().values
    cov_matrix = train_returns.cov().values
    n = len(mean_returns)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0, 1) for _ in range(n))
    initial = np.array([1 / n] * n)

    def objective(weights):
        expected_return = np.dot(weights, mean_returns) * 252
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        if volatility <= 0:
            return 1e6
        sharpe = (expected_return - risk_free_rate) / volatility
        return -sharpe

    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x if result.success else initial


def _annualized_sharpe(daily_returns, risk_free_rate):
    if len(daily_returns) < 2:
        return np.nan
    excess = daily_returns - (risk_free_rate / 252)
    denom = excess.std()
    if denom == 0:
        return np.nan
    return np.sqrt(252) * (excess.mean() / denom)


def run_walk_forward_backtest(
    prices_df,
    risk_free_rate,
    train_days=252,
    rebalance_frequency="M",
    transaction_cost_bps=10,
    output_dir="analysis",
):
    returns = prices_df.pct_change().dropna()
    dates = returns.index
    tickers = returns.columns

    if len(returns) <= train_days + 20:
        raise ValueError("Not enough history for walk-forward backtest.")

    if rebalance_frequency == "M":
        rebalance_frequency = "ME"
    elif rebalance_frequency == "Q":
        rebalance_frequency = "QE"
    rebalance_dates = dates.to_series().resample(rebalance_frequency).last().dropna()
    rebalance_dates = [d for d in rebalance_dates if d in dates and dates.get_loc(d) >= train_days]
    if not rebalance_dates:
        raise ValueError("No valid rebalance dates generated. Check frequency/history.")

    current_weights = np.array([1 / len(tickers)] * len(tickers))
    strategy_nav = []
    benchmark_nav = []
    daily_rows = []
    rebalance_rows = []

    strat_value = 100.0
    bench_value = 100.0
    equal_weights = np.array([1 / len(tickers)] * len(tickers))
    transaction_cost_rate = transaction_cost_bps / 10000.0
    rebalance_set = set(rebalance_dates)

    for date in dates:
        i = dates.get_loc(date)
        if date in rebalance_set:
            train_slice = returns.iloc[i - train_days : i]
            target_weights = _max_sharpe_weights(train_slice, risk_free_rate)
            turnover = np.abs(target_weights - current_weights).sum()
            cost = turnover * transaction_cost_rate * strat_value
            strat_value -= cost
            rebalance_rows.append(
                {
                    "date": date,
                    "turnover": turnover,
                    "transaction_cost_paid": cost,
                    "train_start": train_slice.index.min(),
                    "train_end": train_slice.index.max(),
                }
            )
            current_weights = target_weights

        day_ret_vector = returns.loc[date].values
        strat_daily = float(np.dot(current_weights, day_ret_vector))
        bench_daily = float(np.dot(equal_weights, day_ret_vector))

        strat_value *= 1 + strat_daily
        bench_value *= 1 + bench_daily
        strategy_nav.append(strat_value)
        benchmark_nav.append(bench_value)

        daily_rows.append(
            {
                "date": date,
                "strategy_daily_return": strat_daily,
                "benchmark_daily_return": bench_daily,
                "strategy_nav": strat_value,
                "benchmark_nav": bench_value,
            }
        )

    timeseries = pd.DataFrame(daily_rows)
    strategy_returns = timeseries["strategy_daily_return"]
    benchmark_returns = timeseries["benchmark_daily_return"]
    total_days = len(timeseries)

    summary = pd.DataFrame(
        [
            {
                "metric": "total_return_pct",
                "strategy": (timeseries["strategy_nav"].iloc[-1] / 100 - 1) * 100,
                "benchmark": (timeseries["benchmark_nav"].iloc[-1] / 100 - 1) * 100,
            },
            {
                "metric": "annualized_return_pct",
                "strategy": ((timeseries["strategy_nav"].iloc[-1] / 100) ** (252 / total_days) - 1)
                * 100,
                "benchmark": ((timeseries["benchmark_nav"].iloc[-1] / 100) ** (252 / total_days) - 1)
                * 100,
            },
            {
                "metric": "annualized_volatility_pct",
                "strategy": strategy_returns.std() * np.sqrt(252) * 100,
                "benchmark": benchmark_returns.std() * np.sqrt(252) * 100,
            },
            {
                "metric": "sharpe_ratio",
                "strategy": _annualized_sharpe(strategy_returns, risk_free_rate),
                "benchmark": _annualized_sharpe(benchmark_returns, risk_free_rate),
            },
            {
                "metric": "transaction_cost_total",
                "strategy": sum(r["transaction_cost_paid"] for r in rebalance_rows),
                "benchmark": 0.0,
            },
        ]
    )

    rebalance_log = pd.DataFrame(rebalance_rows)
    os.makedirs(output_dir, exist_ok=True)
    timeseries.to_csv(os.path.join(output_dir, "backtest_timeseries.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "backtest_summary.csv"), index=False)
    rebalance_log.to_csv(os.path.join(output_dir, "rebalance_log.csv"), index=False)

    return timeseries, summary, rebalance_log
