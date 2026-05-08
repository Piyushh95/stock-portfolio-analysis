# Stock Portfolio Optimization and Analysis

End-to-end data science project that analyzes equity returns, optimizes portfolio allocation, and generates reporting artifacts (CSV, charts, and SQL/ML insights).

## What this project includes

- Historical market data collection (India + optional US tech)
- Statistical return/risk analysis and Sharpe-ratio based ranking
- Portfolio optimization using constrained numerical optimization
- Dashboard-ready exports and static visualizations
- SQL-based exploratory analysis with DuckDB
- Basic ML experimentation for next-day return prediction
- Data quality validation checks

## Quick start

```bash
pip install -r requirements.txt
python 2_fetch_nifty50.py
python 3_analysis.py
python 4_portfolio_optimization.py
python 5_prepare_powerbi_data.py
python scripts/5_sql_analysis.py
python scripts/6_ml_predictions.py
python scripts/7_data_validation.py
```

## Project structure

```text
stock-portfolio-analysis/
├── README.md
├── requirements.txt
├── 1.py
├── 2_fetch_nifty50.py
├── 3_analysis.py
├── 4_portfolio_optimization.py
├── 5_prepare_powerbi_data.py
├── app.py
├── risk_free_rate.py
├── scripts/
│   ├── 5_sql_analysis.py
│   ├── 6_ml_predictions.py
│   └── 7_data_validation.py
├── analysis_modules/
├── data_sources/
├── data/
├── analysis/
└── files/README.md
```

For detailed methodology, results, and narrative documentation, see `files/README.md`.
