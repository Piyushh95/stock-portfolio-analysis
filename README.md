# 📈 Stock Portfolio Research Dashboard

An interactive, end-to-end portfolio analysis platform built with Python and Streamlit. Fetches live equity data, runs risk-return analysis, optimizes portfolio allocation, and provides a full research dashboard — including backtesting, efficient frontier visualization, and factor attribution.

Built on **Nifty 50** (India) with optional **US Tech** stock coverage.

---

## 🖥️ Dashboard Preview <img width="1920" height="1080" alt="Stock portfolio anlysis" src="https://github.com/user-attachments/assets/9dea7921-5957-4e35-911e-ebb64193a424" />


<img width="1920" height="1080" alt="stock porfolio anlysis2" src="https://github.com/user-attachments/assets/8d92325a-b5e3-44c3-8527-0482665b21e9" />


> Live dashboard running at `localhost:8501`

**Tabs available:**
`Stock Search` · `Portfolio Input` · `Overview` · `Efficient Frontier` · `Backtest` · `Factor Attribution`

**Controls:**
- Date range selector (Start / End)
- Rebalance Frequency (Monthly / Quarterly / Annual)
- Transaction Cost slider (basis points)
- Include US Tech toggle
- Risk-Free Rate override (%)

---

## 🚀 What This Project Does

Most retail investors pick stocks by gut feel. This project replaces that with data — fetching real market prices, calculating risk-adjusted returns, optimizing portfolio weights, and backtesting the strategy with realistic transaction costs.

**Pipeline:**
```
Fetch Market Data → Risk/Return Analysis → Portfolio Optimization → Backtest → Power BI Export
```

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| Data Collection | `yfinance`, Python |
| Analysis | `Pandas`, `NumPy`, `Scipy` |
| SQL Exploration | `DuckDB` |
| ML Predictions | `Scikit-learn` |
| Dashboard | `Streamlit` |
| Visualization | `Matplotlib`, `Seaborn`, `Power BI` |
| Data Quality | Custom validation scripts |

---

## 🔑 Key Features

- **Live data ingestion** — fetches real OHLCV data for Nifty 50 constituents + optional US Tech
- **Price chart with moving averages** — MA20 and MA50 overlaid on close price (e.g. AAPL: 143.54% period return)
- **Sharpe Ratio ranking** — ranks stocks by risk-adjusted return to identify top performers
- **Efficient Frontier** — plots the risk-return tradeoff across thousands of random portfolios
- **Portfolio optimization** — constrained numerical optimization (`scipy.optimize`) to maximize Sharpe Ratio
- **Backtesting** — tests the optimized portfolio over historical data with configurable rebalance frequency and transaction costs (in bps)
- **Factor Attribution** — breaks down portfolio returns by contributing factors
- **SQL-based EDA** — DuckDB queries for exploratory analysis without a database server
- **ML return prediction** — baseline next-day return prediction model
- **Data validation** — automated checks for missing data, outliers, and quality issues
- **Power BI ready** — exports clean CSVs formatted for direct dashboard consumption

---

## 📊 Example Output

- AAPL Latest Close: **$299.52** | Period Return: **+143.54%** (Jan 2023 – May 2026)
- MA20 / MA50 crossover visualized on interactive chart
- Optimized portfolio weights with expected return and volatility
- Downloadable stock data CSV per ticker

---

## 🗂️ Project Structure

```
stock-portfolio-analysis/
├── 2_fetch_nifty50.py          # Data collection from yfinance
├── 3_analysis.py               # Return/risk analysis + Sharpe ranking
├── 4_portfolio_optimization.py # Constrained optimization (max Sharpe)
├── 5_prepare_powerbi_data.py   # Export pipeline for Power BI
├── app.py                      # Streamlit dashboard (multi-tab)
├── risk_free_rate.py           # Risk-free rate utilities
├── scripts/
│   ├── 5_sql_analysis.py       # DuckDB SQL exploration
│   ├── 6_ml_predictions.py     # Next-day return prediction
│   └── 7_data_validation.py    # Data quality checks
├── analysis_modules/           # Modular analysis components
├── data/                       # Raw and processed data
├── analysis/                   # Output charts and reports
└── files/                      # Detailed methodology docs
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt

# Run full pipeline
python 2_fetch_nifty50.py
python 3_analysis.py
python 4_portfolio_optimization.py
python 5_prepare_powerbi_data.py

# Launch dashboard
streamlit run app.py

# Optional: SQL analysis + ML
python scripts/5_sql_analysis.py
python scripts/6_ml_predictions.py
python scripts/7_data_validation.py
```

---

## 💡 What I'd Improve Next

- Real-time price streaming (WebSocket) instead of batch fetch
- LSTM model for time-series return forecasting
- Portfolio vs Nifty 50 benchmark comparison chart
- User authentication + saved portfolios
- Cloud deployment (Streamlit Cloud / Railway)

---

## ⚠️ Disclaimer

This project is for educational purposes only and is not financial advice.

