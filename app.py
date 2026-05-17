from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


ANALYSIS_DIR = Path("analysis")


def read_csv_if_exists(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def normalize_weights(raw_weights: list[float]) -> list[float]:
    total = float(np.sum(raw_weights))
    if total <= 0:
        return [0.0 for _ in raw_weights]
    return [w / total for w in raw_weights]


st.set_page_config(page_title="Portfolio Research Dashboard", layout="wide")
st.title("Portfolio Research Dashboard")

with st.sidebar:
    st.header("Controls")
    start_date = st.date_input("Start Date", value=pd.Timestamp.today() - pd.Timedelta(days=730))
    end_date = st.date_input("End Date", value=pd.Timestamp.today())
    rebalance_freq = st.selectbox("Rebalance Frequency", ["Monthly", "Quarterly"])
    transaction_cost_bps = st.slider("Transaction Cost (bps)", 0, 100, 10, step=1)
    include_us = st.toggle("Include US Tech", value=True)
    rf_override = st.number_input(
        "Risk-Free Override (%)", value=6.5, min_value=0.0, max_value=20.0, step=0.1
    )
    st.caption(
        "These controls are applied by `run_pipeline.py` when you use it; this dashboard "
        "visualizes the latest generated outputs in `analysis/`."
    )

tabs = st.tabs(
    [
        "Stock Search",
        "Portfolio Input",
        "Overview",
        "Efficient Frontier",
        "Backtest",
        "Factor Attribution",
    ]
)

with tabs[0]:
    st.subheader("Stock Search")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        ticker = st.text_input("Ticker Symbol", value="AAPL").strip().upper()
    with c2:
        stock_start = st.date_input("Start", value=pd.Timestamp("2023-01-01"), key="stock_start")
    with c3:
        stock_end = st.date_input("End", value=pd.Timestamp.today(), key="stock_end")

    if ticker:
        stock_df = yf.download(ticker, start=stock_start, end=stock_end, progress=False)
        if stock_df.empty:
            st.warning("No data returned for this ticker/date range.")
        else:
            st.subheader("Recent Data")
            st.dataframe(stock_df.tail(10), use_container_width=True)

            close_col = "Close"
            if isinstance(stock_df.columns, pd.MultiIndex):
                close_col = ("Close", ticker)
            close_series = stock_df[close_col]

            ma20 = close_series.rolling(20).mean()
            ma50 = close_series.rolling(50).mean()

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(close_series.index, close_series, label="Close", linewidth=1.6)
            ax.plot(ma20.index, ma20, label="MA 20", linewidth=1.2)
            ax.plot(ma50.index, ma50, label="MA 50", linewidth=1.2)
            ax.set_title(f"{ticker} Price and Moving Averages")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            latest_close = float(close_series.iloc[-1])
            change_pct = float((close_series.iloc[-1] / close_series.iloc[0] - 1) * 100)
            m1, m2 = st.columns(2)
            m1.metric("Latest Close", f"{latest_close:.2f}")
            m2.metric("Period Return", f"{change_pct:.2f}%")

            csv_bytes = stock_df.to_csv().encode("utf-8")
            st.download_button(
                "Download Stock Data CSV",
                data=csv_bytes,
                file_name=f"{ticker.lower()}_prices.csv",
                mime="text/csv",
            )

with tabs[1]:
    st.subheader("Portfolio Input")
    st.caption("Enter tickers and target weights. Weights are normalized to 100%.")

    symbols_raw = st.text_input("Portfolio Symbols (comma-separated)", "AAPL,MSFT,GOOGL")
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]

    if not symbols:
        st.info("Add at least one symbol to build a portfolio.")
    else:
        raw_weights = []
        for sym in symbols:
            raw_weights.append(st.slider(f"{sym} weight", min_value=0.0, max_value=100.0, value=10.0, step=0.5))

        norm_weights = normalize_weights(raw_weights)
        portfolio_df = pd.DataFrame(
            {
                "Stock": symbols,
                "Input Weight": raw_weights,
                "Normalized Weight (%)": [w * 100 for w in norm_weights],
            }
        )
        st.dataframe(portfolio_df, use_container_width=True)

        pie = px.pie(portfolio_df, values="Normalized Weight (%)", names="Stock", title="Portfolio Allocation")
        st.plotly_chart(pie, use_container_width=True)

        returns_ts = read_csv_if_exists(ANALYSIS_DIR / "returns_timeseries.csv")
        if returns_ts is not None:
            aligned = [s for s in symbols if s in returns_ts.columns]
            if aligned:
                tmp = returns_ts[aligned].copy()
                aligned_weights = np.array(
                    [norm_weights[symbols.index(s)] for s in aligned],
                    dtype=float,
                )
                weighted_returns = (tmp * aligned_weights).sum(axis=1)
                est_annual_return = float(weighted_returns.mean() * 252 * 100)
                est_annual_vol = float(weighted_returns.std() * np.sqrt(252) * 100)
                c1, c2 = st.columns(2)
                c1.metric("Estimated Annual Return", f"{est_annual_return:.2f}%")
                c2.metric("Estimated Annual Volatility", f"{est_annual_vol:.2f}%")
            else:
                st.info("No matching symbols found in `analysis/returns_timeseries.csv` for risk metrics.")

with tabs[2]:
    st.subheader("Overview")
    summary = read_csv_if_exists(ANALYSIS_DIR / "stock_summary.csv")
    weights = read_csv_if_exists(ANALYSIS_DIR / "portfolio_weights.csv")
    perf = read_csv_if_exists(ANALYSIS_DIR / "portfolio_performance.csv")
    universe_meta = read_csv_if_exists(ANALYSIS_DIR / "universe_metadata.csv")

    if summary is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(summary.sort_values("Sharpe Ratio", ascending=False), use_container_width=True)
        with c2:
            fig = px.scatter(
                summary,
                x="Annual Volatility (%)",
                y="Annual Return (%)",
                text="Stock",
                title="Risk vs Return",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing `analysis/stock_summary.csv`. Run pipeline first.")

    if weights is not None:
        fig = px.pie(weights[weights["Weight (%)"] > 0.01], values="Weight (%)", names="Stock")
        fig.update_layout(title="Current Portfolio Allocation")
        st.plotly_chart(fig, use_container_width=True)

    if perf is not None:
        fig = px.line(perf, x="Date", y="Portfolio_Cumulative_Return", title="Portfolio Cumulative Return")
        st.plotly_chart(fig, use_container_width=True)

    if universe_meta is not None:
        st.caption("Universe Metadata")
        st.dataframe(universe_meta.head(20), use_container_width=True)

with tabs[3]:
    st.subheader("Efficient Frontier")
    frontier = read_csv_if_exists(ANALYSIS_DIR / "efficient_frontier.csv")
    if frontier is None:
        st.info("No frontier output yet. Generate `analysis/efficient_frontier.csv`.")
    else:
        fig = px.scatter(
            frontier,
            x="volatility",
            y="expected_return",
            color="sharpe",
            title="Efficient Frontier",
            labels={"volatility": "Volatility", "expected_return": "Expected Return"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(frontier.sort_values("sharpe", ascending=False).head(20), use_container_width=True)

with tabs[4]:
    st.subheader("Backtest")
    ts = read_csv_if_exists(ANALYSIS_DIR / "backtest_timeseries.csv")
    summary = read_csv_if_exists(ANALYSIS_DIR / "backtest_summary.csv")
    rebalance = read_csv_if_exists(ANALYSIS_DIR / "rebalance_log.csv")

    if ts is None:
        st.info("No backtest output yet. Generate `analysis/backtest_timeseries.csv`.")
    else:
        fig = px.line(ts, x="date", y=["strategy_nav", "benchmark_nav"], title="Strategy vs Benchmark NAV")
        st.plotly_chart(fig, use_container_width=True)

    if summary is not None:
        st.dataframe(summary, use_container_width=True)

    if rebalance is not None:
        st.caption("Rebalance Log")
        st.dataframe(rebalance.tail(50), use_container_width=True)

with tabs[5]:
    st.subheader("Factor Attribution")
    exposures = read_csv_if_exists(ANALYSIS_DIR / "factor_exposures.csv")
    contrib = read_csv_if_exists(ANALYSIS_DIR / "factor_contributions.csv")

    if exposures is None:
        st.info("No factor outputs yet. Generate factor analysis files.")
    else:
        fig = px.bar(exposures, x="factor", y="beta", title="Factor Exposures")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(exposures, use_container_width=True)

    if contrib is not None:
        fig = px.bar(contrib, x="factor", y="contribution", title="Factor Contribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(contrib, use_container_width=True)
