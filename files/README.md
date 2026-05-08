# Stock Portfolio Optimization & Analysis

> **Data Science Project**: Analyzing 21 stocks across Nifty 50 and US markets to find optimal portfolio allocation using mathematical optimization.

## 🎯 Project Overview

This project demonstrates a **complete data analytics pipeline**: from data collection and cleaning through statistical analysis to optimization and visualization. It answers a real-world question: *"How do you build a portfolio that maximizes risk-adjusted returns?"*

### Key Achievement
**924% improvement in risk-adjusted returns** through data-driven portfolio optimization
- Equal-weight allocation: Sharpe ratio 0.14 (Annual return 6.58%, Risk 11.68%)
- Optimal allocation: Sharpe ratio 1.39 (Annual return 31.34%, Risk 19.00%)

---

## 📊 Results Summary

### Optimal Portfolio Allocation
| Stock | Allocation | Reason |
|-------|-----------|--------|
| **GOOGL** | 53.96% | Highest Sharpe ratio (1.25), best risk-adjusted returns |
| **JSWSTEEL.NS** | 24.96% | Low correlation with tech, sector diversification |
| **ICICIBANK.NS** | 11.36% | Banking sector exposure, stability |
| **AAPL** | 6.13% | US market diversification, lower volatility |
| **AXISBANK.NS** | 3.60% | Additional banking sector balance |

### Stock Performance Analysis
**Best Performers:**
- GOOGL: +184% return, 1.25 Sharpe ratio
- AAPL: +51% return, 0.92 Sharpe ratio  
- MSFT: +26% return, 0.65 Sharpe ratio

**Worst Performers:**
- TCS: -31% return, -0.98 Sharpe ratio
- RELIANCE: -8% return, -0.31 Sharpe ratio
- INFY: -6% return, -0.20 Sharpe ratio

---

## 🛠️ Tech Stack

**Programming & Data Processing:**
- Python 3.8+
- Pandas (data manipulation, cleaning)
- NumPy (numerical computation)
- SciPy (optimization algorithms)

**Visualization:**
- Matplotlib (static charts)
- Seaborn (statistical visualizations)
- Plotly (interactive dashboards)

**Data Source:**
- yfinance (Yahoo Finance API)

**Version Control:**
- Git/GitHub

---

## 📁 Project Structure

```
stock-portfolio-analysis/
│
├── README.md                          # Main project overview (repo root)
├── requirements.txt                   # Dependencies
│
├── data/
│   ├── stock_prices.csv              # Raw 2-year historical data (21 stocks)
│   └── optimal_weights.csv           # Optimal portfolio allocation results
│
├── 1.py                             # Early data fetch prototype
├── 2_fetch_nifty50.py               # Download stock data
├── 3_analysis.py                    # Statistical analysis
├── 4_portfolio_optimization.py      # Find optimal weights
├── 5_prepare_powerbi_data.py        # Prepare visualizations
├── scripts/
│   ├── 5_sql_analysis.py            # SQL analytics with window functions
│   ├── 6_ml_predictions.py          # ML prediction + feature importance
│   └── 7_data_validation.py         # Data quality checks
│
├── analysis/
│   ├── stock_summary.csv            # Key metrics per stock
│   ├── portfolio_performance.csv     # Returns over time
│   ├── price_timeseries.csv         # Daily prices
│   ├── correlation_data.csv         # Stock correlations
│   ├── returns_timeseries.csv       # Daily returns
│   ├── ml_analysis.png              # ML model performance dashboard
│   │
│   ├── portfolio_dashboard.png      # 4-panel visualization
│   ├── stock_performance_lines.png  # Price trends
│   └── allocation_breakdown.png     # Allocation bar chart
│
└── dashboard/
    ├── index.html                   # Master interactive dashboard
    ├── portfolio_dashboard.html      # 4-panel overview
    ├── portfolio_comparison.html     # Optimal vs Equal-weight
    ├── stock_performance.html        # Individual stock trends
    ├── portfolio_allocation.html     # Allocation breakdown
    └── stock_metrics.html            # Key metrics table
```

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/stock-portfolio-analysis.git
cd stock-portfolio-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Analysis Pipeline

**Step 1: Download Data** (2-year historical prices)
```bash
python 2_fetch_nifty50.py
```
Output: `data/stock_prices.csv`

**Step 2: Analyze Stocks** (Calculate returns, volatility, Sharpe ratio)
```bash
python 3_analysis.py
```
Output: Key metrics for each stock

**Step 3: Optimize Portfolio** (Find best allocation)
```bash
python 4_portfolio_optimization.py
```
Output: Optimal weights and comparison metrics

**Step 4: Generate Visualizations** (Create charts and dashboards)
```bash
python 5_prepare_powerbi_data.py
```
Output: CSV files for Power BI + PNG charts + Interactive HTML dashboard

**Step 5: SQL Analysis** (Portfolio insights using SQL)
```bash
python scripts/5_sql_analysis.py
```
Output: Ranked stocks, allocation concentration, and risk-return classes

**Step 6: ML Predictions** (Next-day return modeling)
```bash
python scripts/6_ml_predictions.py
```
Output: `analysis/ml_analysis.png` + model metrics (R^2, RMSE, MAE)

**Step 7: Data Validation** (Production-style quality checks)
```bash
python scripts/7_data_validation.py
```
Output: Data validation report (missing values, outliers, jump checks, normality)

### 4. View Results

**Interactive Dashboard:**
```bash
open dashboard/index.html  # macOS
# or
start dashboard/index.html  # Windows
```

**Static Visualizations:**
- `analysis/portfolio_dashboard.png` - 4-panel summary
- `analysis/stock_performance_lines.png` - Price trends
- `analysis/allocation_breakdown.png` - Allocation chart

---

## 📈 Methodology

### 1. Data Collection & Cleaning
- Downloaded 2-year historical daily prices for 21 stocks
- Cleaned missing values using forward-fill
- Validated data quality and consistency

### 2. Statistical Analysis
Calculated key financial metrics for each stock:
- **Daily Returns**: `(Price_t - Price_t-1) / Price_t-1`
- **Annual Return**: Mean daily return × 252 (trading days/year)
- **Volatility**: Std dev of daily returns × √252
- **Sharpe Ratio**: `(Annual Return - Risk-free rate) / Volatility`
  - Risk-free rate assumed at 5%
- **Correlation Matrix**: Pairwise correlations between stocks

### 3. Diversification Analysis
- Low correlation between US tech (AAPL, MSFT, GOOGL) and Indian stocks → good diversification
- High correlation between INFY and TCS (0.74) → don't hold both
- RELIANCE low correlation → good hedge against tech exposure

### 4. Portfolio Optimization
**Objective**: Maximize Sharpe Ratio (risk-adjusted returns)

**Constraints**:
- Weights sum to 100% (fully invested)
- No short selling (weights ≥ 0)
- No leverage (weights ≤ 100%)

**Algorithm**: Sequential Least Squares Programming (SLSQP) via SciPy

**Results**:
```
Optimal: GOOGL (54%), JSWSTEEL (25%), ICICIBANK (11%), AAPL (6%), AXISBANK (4%)
Performance: 31.34% return, 1.39 Sharpe ratio

vs

Equal-weight: 4.76% each stock
Performance: 6.58% return, 0.14 Sharpe ratio

Improvement: +924% on Sharpe ratio
```

### 5. Visualization & Storytelling
- 4-panel dashboard combining price trends, correlation, risk-return, allocation
- Interactive HTML dashboards for exploration
- Static PNG charts for presentations

### 6. SQL Analysis
- Uses DuckDB to run analytical SQL directly on generated CSV outputs
- Demonstrates window functions: `RANK()`, `NTILE()`, and aggregate-over-window logic
- Classifies stocks into risk-return buckets for easier portfolio interpretation

### 7. Machine Learning
- Builds predictive models for next-day returns (Linear Regression and Random Forest)
- Performs feature engineering with lag returns, rolling volatility, momentum, and EMA
- Compares models using R^2, RMSE, and MAE and exports an ML analysis chart

### 8. Data Validation
- Checks missing values, duplicate timestamps, invalid prices, and extreme jumps
- Detects outliers using z-score logic and validates returns boundaries
- Includes a compact quality score summary for quick production-readiness checks

---

## 📊 Key Insights

### 1. Data-Driven Beats Naive
Equal-weight allocation spreads capital equally across all stocks, including underperformers (TCS -31%). Optimization concentrates in winners while maintaining diversification.

### 2. Sector Balance Matters
- Pure tech portfolio (GOOGL, AAPL, MSFT) = good returns, high correlation
- Adding steel (JSWSTEEL) + banking (ICICIBANK) = lower correlation, better risk management
- US + India mix = cross-market diversification

### 3. Correlation is Key
- High correlation (INFY-TCS: 0.74) = redundant diversification
- Low correlation (RELIANCE-GOOGL: 0.08) = effective hedge
- Optimal portfolio exploits these relationships

### 4. Risk-Adjusted > Absolute Returns
- Simple "pick the highest return stock" approach = TSLA (58% annual return)
- But TSLA has 61% volatility → risky
- Sharpe ratio shows GOOGL (1.25) better than TSLA (0.86) on risk-adjusted basis

---

## 🔍 Analysis Details

### Stocks Analyzed (21 Total)

**Indian Banks (4)**
- HDFCBANK.NS, ICICIBANK.NS, AXISBANK.NS, KOTAK.NS

**IT/Software (4)**
- TCS.NS, INFY.NS, WIPRO.NS, LTTS.NS

**Energy (2)**
- RELIANCE.NS, POWERGRID.NS

**Automotive (2)**
- MARUTI.NS, TATAMOTORS.NS

**Pharma (2)**
- SUNPHARMA.NS, DRREDDY.NS

**FMCG (3)**
- ITC.NS, HINDUNILVR.NS, NESTLEIND.NS

**Cement (2)**
- ULTRAMARINE.NS, SHREECEM.NS

**Steel (2)**
- TATASTEEL.NS, JSWSTEEL.NS

**US Tech (3)**
- AAPL, MSFT, GOOGL

---

## 💡 What I Learned

### Technical Skills
✅ Data pipeline: extraction → cleaning → analysis → optimization → visualization  
✅ Statistical analysis: returns, volatility, correlation, Sharpe ratio  
✅ Mathematical optimization: constraint-based optimization with SciPy  
✅ Data visualization: matplotlib, seaborn, plotly  
✅ Python best practices: pandas operations, efficient data processing  

### Data Science Concepts
✅ Risk-return tradeoff and Sharpe ratio  
✅ Diversification and correlation  
✅ Portfolio theory and optimization  
✅ Exploratory data analysis (EDA)  
✅ Comparative analysis (benchmark vs optimized)  

### Problem-Solving
✅ Breaking down complex problem into steps  
✅ Comparing naive vs sophisticated approaches  
✅ Identifying key insights from data  
✅ Communicating findings visually  

---

## 🚧 Limitations & Future Work

### Current Limitations
1. **Historical Data**: Analysis based on 2-year period; markets change
2. **Static Optimization**: Doesn't account for market regime changes
3. **Transaction Costs**: Ignores fees and slippage
4. **Rebalancing**: Optimal weights don't change; real portfolios need periodic rebalancing
5. **Risk-Free Rate**: Assumed static 5%; should use real government bond rates

### Future Enhancements
1. **Rolling Window Optimization**: Recalculate optimal weights quarterly/yearly
2. **Sector-Level Constraints**: Limit exposure to any single sector
3. **Minimum Diversification**: Enforce minimum holdings to reduce concentration
4. **Transaction Cost Analysis**: Model impact of trading fees
5. **Monte Carlo Simulation**: Generate forward-looking return distributions
6. **Stress Testing**: Analyze portfolio performance in extreme scenarios
7. **Risk Parity**: Alternative approach weighting stocks by inverse volatility
8. **Machine Learning**: Predict returns using feature engineering and regression

### Possible Extensions
- [ ] Add real-time data updates
- [ ] Implement portfolio rebalancing alerts
- [ ] Build web app (Flask/Django) for interactive optimization
- [ ] Add more assets (bonds, commodities, cryptocurrencies)
- [ ] Implement multi-period optimization (dynamic programming)
- [ ] Add performance attribution analysis

---

## 📚 Learning Resources

### Concepts Used
- Portfolio Theory: Markowitz Modern Portfolio Theory
- Risk Metrics: Sharpe Ratio, Standard Deviation, VaR
- Optimization: Constrained optimization, SLSQP algorithm
- Statistics: Correlation, probability distributions

### Books & References
- "A Random Walk Down Wall Street" - Burton Malkiel (Portfolio theory basics)
- "Machine Learning for Algorithmic Trading" - Stefan Jansen (Practical applications)

### Python Resources
- Pandas Documentation: https://pandas.pydata.org/
- NumPy Documentation: https://numpy.org/
- SciPy Optimization: https://docs.scipy.org/doc/scipy/reference/optimize.html

---

## 🤝 Contributing

This is a personal project, but feedback is welcome! Feel free to:
- Report bugs or data issues
- Suggest improvements
- Propose new analyses

---

## 📄 License

This project is open source for educational purposes.

---

## 👨‍💻 About

**Author**: Piyush Bhosale  
**Date**: April 2026  
**Purpose**: Data Science Portfolio Project  

**Contact**:
- Email: piyushbhosale3700@gmail.com
- LinkedIn: [Your LinkedIn]
- GitHub: [Your GitHub]

---

## 🔗 Related Projects

- [Stock Price Prediction with ML](#) (Coming soon)
- [Real-Time Portfolio Dashboard](#) (Coming soon)
- [Sentiment Analysis on Market News](#) (Coming soon)

---

**Last Updated**: April 2026
