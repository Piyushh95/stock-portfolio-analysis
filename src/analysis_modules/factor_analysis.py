import os

import numpy as np
import pandas as pd


def _load_or_build_factors(prices_df, factor_file_path):
    """
    Load factor returns from CSV if present, otherwise build proxy factors.
    Expected CSV columns: date, MKT, SMB, MOM, RF (daily decimal returns/rate).
    """
    if os.path.exists(factor_file_path):
        df = pd.read_csv(factor_file_path, parse_dates=["date"]).set_index("date").sort_index()
        required = {"MKT", "SMB", "MOM", "RF"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Factor file missing columns: {sorted(missing)}")
        return df[["MKT", "SMB", "MOM", "RF"]], "user_factor_csv"

    # Proxy fallback when user factor file does not exist.
    returns = prices_df.pct_change().dropna()
    market = returns.mean(axis=1)

    # SMB proxy: small-cap proxy via inverse price rank (imperfect but deterministic fallback).
    last_prices = prices_df.ffill().iloc[-1]
    ranked = last_prices.rank(method="first")
    small = ranked <= ranked.median()
    big = ~small
    smb = returns.loc[:, small.values].mean(axis=1) - returns.loc[:, big.values].mean(axis=1)

    # MOM proxy: long recent winners, short recent losers.
    rolling_20 = returns.rolling(20).mean()
    mom = pd.Series(index=returns.index, dtype=float)
    for dt in returns.index[20:]:
        scores = rolling_20.loc[dt].dropna()
        if len(scores) < 6:
            mom.loc[dt] = np.nan
            continue
        q = len(scores) // 3
        winners = scores.sort_values(ascending=False).index[:q]
        losers = scores.sort_values(ascending=True).index[:q]
        mom.loc[dt] = returns.loc[dt, winners].mean() - returns.loc[dt, losers].mean()

    # RF proxy fallback: fixed annual 5% converted to daily.
    rf_daily = pd.Series(0.05 / 252, index=returns.index)

    factors = pd.DataFrame({"MKT": market, "SMB": smb, "MOM": mom, "RF": rf_daily}).dropna()
    return factors, "proxy_generated"


def _ols(y, X):
    """
    OLS with intercept. Returns beta, stderr, t_stats, residuals.
    """
    X_mat = np.column_stack([np.ones(len(X)), X.values])
    y_vec = y.values.reshape(-1, 1)
    xtx_inv = np.linalg.inv(X_mat.T @ X_mat)
    beta = xtx_inv @ X_mat.T @ y_vec
    residuals = y_vec - X_mat @ beta
    n, k = X_mat.shape
    sigma2 = ((residuals.T @ residuals) / (n - k)).item()
    var_beta = sigma2 * xtx_inv
    stderr = np.sqrt(np.diag(var_beta)).reshape(-1, 1)
    t_stats = beta / stderr
    return beta.flatten(), stderr.flatten(), t_stats.flatten(), residuals.flatten()


def run_factor_analysis(
    prices_df,
    weights_df,
    factor_file_path="data/factors/factor_returns.csv",
    output_dir="analysis",
):
    returns = prices_df.pct_change().dropna()
    w = weights_df.set_index("Stock")["Weight (%)"] / 100
    w = w.reindex(returns.columns).fillna(0)
    portfolio_ret = (returns * w.values).sum(axis=1)

    factors, source = _load_or_build_factors(prices_df, factor_file_path)
    merged = pd.concat([portfolio_ret.rename("portfolio"), factors], axis=1).dropna()

    y = merged["portfolio"] - merged["RF"]
    X = merged[["MKT", "SMB", "MOM"]]
    beta, stderr, t_stats, residuals = _ols(y, X)

    alpha_daily = beta[0]
    r_squared = 1 - (np.var(residuals) / np.var(y.values))
    exposures = pd.DataFrame(
        [
            {"factor": "alpha", "beta": beta[0], "stderr": stderr[0], "t_stat": t_stats[0], "source": source},
            {"factor": "MKT", "beta": beta[1], "stderr": stderr[1], "t_stat": t_stats[1], "source": source},
            {"factor": "SMB", "beta": beta[2], "stderr": stderr[2], "t_stat": t_stats[2], "source": source},
            {"factor": "MOM", "beta": beta[3], "stderr": stderr[3], "t_stat": t_stats[3], "source": source},
            {"factor": "r_squared", "beta": r_squared, "stderr": np.nan, "t_stat": np.nan, "source": source},
        ]
    )

    annual_factor_means = X.mean() * 252
    contributions = pd.DataFrame(
        [
            {"factor": "MKT", "contribution": beta[1] * annual_factor_means["MKT"]},
            {"factor": "SMB", "contribution": beta[2] * annual_factor_means["SMB"]},
            {"factor": "MOM", "contribution": beta[3] * annual_factor_means["MOM"]},
            {"factor": "alpha", "contribution": alpha_daily * 252},
        ]
    )
    contributions["source"] = source

    os.makedirs(output_dir, exist_ok=True)
    exposures.to_csv(os.path.join(output_dir, "factor_exposures.csv"), index=False)
    contributions.to_csv(os.path.join(output_dir, "factor_contributions.csv"), index=False)
    merged.to_csv(os.path.join(output_dir, "factor_merged_data.csv"))

    return exposures, contributions
