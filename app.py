from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ANALYSIS_DIR = Path("analysis")


def read_csv_if_exists(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


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

tabs = st.tabs(["Overview", "Efficient Frontier", "Backtest", "Factor Attribution"])

with tabs[0]:
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

with tabs[1]:
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

with tabs[2]:
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

with tabs[3]:
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
