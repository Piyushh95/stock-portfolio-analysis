import duckdb
import pandas as pd


# Initialize DuckDB connection
conn = duckdb.connect(":memory:")

# Load CSV files as SQL tables
returns_df = pd.read_csv("analysis/returns_timeseries.csv")
summary_df = pd.read_csv("analysis/stock_summary.csv")
weights_df = pd.read_csv("data/optimal_weights.csv")

conn.register("returns_data", returns_df)
conn.register("stock_summary", summary_df)
conn.register("portfolio_weights", weights_df)

print("\n" + "=" * 80)
print("SQL ANALYSIS: Portfolio Data Queries")
print("=" * 80)

# Query 1: Top 5 stocks by Sharpe ratio
query1 = """
    SELECT
        Stock,
        ROUND("Total Return (%)", 2) AS total_return,
        ROUND("Sharpe Ratio", 2) AS sharpe_ratio,
        RANK() OVER (ORDER BY "Sharpe Ratio" DESC) AS rank
    FROM stock_summary
    ORDER BY "Sharpe Ratio" DESC
    LIMIT 5
"""

print("\nQUERY 1: Top 5 Stocks by Sharpe Ratio")
print("-" * 80)
result = conn.execute(query1).fetchall()
for row in result:
    print(
        f"  {row[0]:15s} | Return: {row[1]:7.2f}% | Sharpe: {row[2]:6.2f} | Rank: {row[3]}"
    )

# Query 2: Stocks sorted by return with quartile ranking
query2 = """
    SELECT
        Stock,
        ROUND("Annual Return (%)", 2) AS annual_return,
        NTILE(4) OVER (ORDER BY "Annual Return (%)") AS quartile,
        CASE
            WHEN "Annual Return (%)" > 10 THEN 'High'
            WHEN "Annual Return (%)" > 0 THEN 'Positive'
            ELSE 'Negative'
        END AS performance_category
    FROM stock_summary
    ORDER BY "Annual Return (%)" DESC
"""

print("\nQUERY 2: Stocks by Return with Quartile and Category")
print("-" * 80)
result = conn.execute(query2).fetchall()
for row in result:
    print(f"  {row[0]:15s} | Return: {row[1]:7.2f}% | Q{row[2]} | {row[3]:8s}")

# Query 3: Portfolio weight analysis
query3 = """
    SELECT
        Stock,
        ROUND("Weight (%)", 2) AS weight,
        ROUND("Weight (%)" / SUM("Weight (%)") OVER () * 100, 2) AS weight_pct,
        CASE
            WHEN "Weight (%)" > 20 THEN 'High Concentration'
            WHEN "Weight (%)" > 5 THEN 'Medium Concentration'
            WHEN "Weight (%)" > 0 THEN 'Low Concentration'
            ELSE 'Not Held'
        END AS concentration_level
    FROM portfolio_weights
    WHERE "Weight (%)" > 0.01
    ORDER BY "Weight (%)" DESC
"""

print("\nQUERY 3: Portfolio Allocation Analysis")
print("-" * 80)
result = conn.execute(query3).fetchall()
for row in result:
    print(
        f"  {row[0]:15s} | Weight: {row[1]:6.2f}% | Pct: {row[2]:6.2f}% | {row[3]}"
    )

# Query 4: Risk-return matrix
query4 = """
    SELECT
        Stock,
        CASE
            WHEN "Annual Return (%)" > 20 AND "Annual Volatility (%)" < 30 THEN 'High Return, Low Risk'
            WHEN "Annual Return (%)" > 20 THEN 'High Return, High Risk'
            WHEN "Annual Return (%)" > 0 AND "Annual Volatility (%)" < 30 THEN 'Moderate Return, Low Risk'
            WHEN "Annual Return (%)" > 0 THEN 'Moderate Return, High Risk'
            ELSE 'Negative Return'
        END AS risk_return_category,
        ROUND("Annual Return (%)", 2) AS ret,
        ROUND("Annual Volatility (%)", 2) AS vol
    FROM stock_summary
    ORDER BY "Annual Return (%)" DESC
"""

print("\nQUERY 4: Risk-Return Matrix Classification")
print("-" * 80)
result = conn.execute(query4).fetchall()
for row in result:
    print(
        f"  {row[0]:15s} | {row[1]:30s} | Return: {row[2]:7.2f}% | Vol: {row[3]:6.2f}%"
    )

# Query 5: Portfolio performance statistics
query5 = """
    SELECT
        'Optimal Portfolio' AS portfolio_type,
        COUNT(DISTINCT pw.Stock) AS num_stocks,
        SUM("Weight (%)") AS total_weight,
        ROUND(AVG("Annual Return (%)"), 2) AS avg_return,
        ROUND(STDDEV_POP("Annual Return (%)"), 2) AS return_stddev,
        ROUND(MAX("Sharpe Ratio"), 2) AS best_sharpe,
        ROUND(MIN("Sharpe Ratio"), 2) AS worst_sharpe
    FROM portfolio_weights pw
    JOIN stock_summary ss ON pw.Stock = ss.Stock
    WHERE pw."Weight (%)" > 0.01
"""

print("\nQUERY 5: Optimal Portfolio Statistics")
print("-" * 80)
result = conn.execute(query5).fetchall()
for row in result:
    print(f"  Portfolio Type: {row[0]}")
    print(f"  Number of Stocks: {row[1]}")
    print(f"  Total Weight: {row[2]:.2f}%")
    print(f"  Avg Return: {row[3]:.2f}%")
    print(f"  Return Std Dev: {row[4]:.2f}%")
    print(f"  Best Sharpe: {row[5]:.2f}")
    print(f"  Worst Sharpe: {row[6]:.2f}")

print("\n" + "=" * 80)
print("SQL Analysis Complete")
print("=" * 80)
