import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create analysis folder
os.makedirs('../analysis', exist_ok=True)

# Load data
data = pd.read_csv('../data/stock_prices.csv', index_col=0, parse_dates=True)
daily_returns = data.pct_change().dropna()
correlation_matrix = daily_returns.corr()
weights_df = pd.read_csv('../data/optimal_weights.csv')

print("Preparing Power BI data files...\n")

# ===== 1. TIME SERIES DATA =====
price_timeseries = data.reset_index()
price_timeseries.to_csv('../analysis/price_timeseries.csv', index=False)
print("price_timeseries.csv")

# ===== 2. RETURNS TIME SERIES =====
returns_timeseries = daily_returns.reset_index()
returns_timeseries.to_csv('../analysis/returns_timeseries.csv', index=False)
print("returns_timeseries.csv")

# ===== 3. CORRELATION MATRIX (melted for Power BI) =====
corr_melted = correlation_matrix.reset_index().melt(id_vars='index')
corr_melted.columns = ['Stock1', 'Stock2', 'Correlation']
corr_melted.to_csv('../analysis/correlation_data.csv', index=False)
print("correlation_data.csv")

# ===== 4. STOCK SUMMARY =====
stock_summary = pd.read_csv('../analysis/stock_summary.csv')
stock_summary.to_csv('../analysis/stock_summary.csv', index=False)
print("stock_summary.csv")

# ===== 5. PORTFOLIO PERFORMANCE =====
weights_array = weights_df.set_index('Stock').loc[data.columns, 'Weight (%)'].values / 100
portfolio_daily_returns = (daily_returns * weights_array).sum(axis=1)
portfolio_cumulative = (1 + portfolio_daily_returns).cumprod() * 100

portfolio_perf = pd.DataFrame({
    'Date': portfolio_daily_returns.index,
    'Portfolio_Cumulative_Return': portfolio_cumulative.values,
    'Portfolio_Daily_Return': (portfolio_daily_returns * 100).values
})
portfolio_perf.to_csv('../analysis/portfolio_performance.csv', index=False)
print("portfolio_performance.csv")

# ===== 6. PORTFOLIO WEIGHTS =====
weights_df.to_csv('../analysis/portfolio_weights.csv', index=False)
print("portfolio_weights.csv")

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80 + "\n")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# ===== VISUALIZATION 1: 4-PANEL DASHBOARD =====
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Panel 1: Normalized Price Trends
normalized_data = (data / data.iloc[0]) * 100
normalized_data.plot(ax=axes[0, 0], linewidth=2, alpha=0.7)
axes[0, 0].set_title('Stock Performance (Normalized to 100)', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Index (Base = 100)', fontsize=11)
axes[0, 0].legend(loc='best', fontsize=8, ncol=2)
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Correlation Heatmap
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm',
            cbar_kws={'label': 'Correlation'}, ax=axes[0, 1], vmin=-1, vmax=1)
axes[0, 1].set_title('Stock Correlation Matrix', fontsize=14, fontweight='bold')

# Panel 3: Risk vs Return Scatter
annual_returns = daily_returns.mean() * 252 * 100
annual_volatility = daily_returns.std() * np.sqrt(252) * 100

ax = axes[1, 0]
colors = ['red' if ret < 0 else 'green' for ret in annual_returns]
ax.scatter(annual_volatility, annual_returns, s=300, alpha=0.6, c=colors, edgecolors='black', linewidth=2)

for stock in daily_returns.columns:
    ax.annotate(stock,
                (annual_volatility[stock], annual_returns[stock]),
                fontsize=9, fontweight='bold',
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Annual Volatility (Risk %)', fontsize=11, fontweight='bold')
ax.set_ylabel('Annual Return (%)', fontsize=11, fontweight='bold')
ax.set_title('Risk vs Return Trade-off', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 4: Optimal Portfolio Allocation
weights_sorted = weights_df.sort_values('Weight (%)', ascending=False)
weights_sorted = weights_sorted[weights_sorted['Weight (%)'] > 0.1]

colors_pie = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
ax = axes[1, 1]
wedges, texts, autotexts = ax.pie(weights_sorted['Weight (%)'],
                                    labels=weights_sorted['Stock'],
                                    autopct='%1.1f%%',
                                    colors=colors_pie[:len(weights_sorted)],
                                    startangle=90,
                                    textprops={'fontsize': 11, 'fontweight': 'bold'})
ax.set_title('Optimal Portfolio Allocation', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../analysis/portfolio_dashboard.png', dpi=300, bbox_inches='tight')
print("portfolio_dashboard.png")
plt.close()

# ===== VISUALIZATION 2: INDIVIDUAL STOCK PERFORMANCE =====
fig, ax = plt.subplots(figsize=(14, 6))
for col in data.columns:
    normalized = (data[col] / data[col].iloc[0]) * 100
    ax.plot(normalized.index, normalized.values, label=col, linewidth=2, alpha=0.7)

ax.set_title('Individual Stock Performance (2-Year Period)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Index (Base = 100)', fontsize=11)
ax.legend(loc='best', fontsize=9, ncol=3)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../analysis/stock_performance_lines.png', dpi=300, bbox_inches='tight')
print("stock_performance_lines.png")
plt.close()

# ===== VISUALIZATION 3: ALLOCATION BAR CHART =====
fig, ax = plt.subplots(figsize=(12, 6))
weights_sorted_all = weights_df.sort_values('Weight (%)', ascending=True)
colors_bar = ['#2ecc71' if w > 0 else '#e74c3c' for w in weights_sorted_all['Weight (%)']]
ax.barh(weights_sorted_all['Stock'], weights_sorted_all['Weight (%)'], color=colors_bar, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Weight (%)', fontsize=11, fontweight='bold')
ax.set_title('Optimal Portfolio Allocation - All Stocks', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add percentage labels
for i, v in enumerate(weights_sorted_all['Weight (%)']):
    ax.text(v + 0.5, i, f'{v:.2f}%', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('../analysis/allocation_breakdown.png', dpi=300, bbox_inches='tight')
print("allocation_breakdown.png")
plt.close()

print("\n" + "="*80)
print("ALL FILES READY FOR POWER BI!")
print("="*80)
print("\nFiles in 'analysis/' folder:")
print("  Data files (for Power BI import):")
print("     - price_timeseries.csv")
print("     - returns_timeseries.csv")
print("     - correlation_data.csv")
print("     - stock_summary.csv")
print("     - portfolio_performance.csv")
print("     - portfolio_weights.csv")
print("\n  Visualizations (reference images):")
print("     - portfolio_dashboard.png")
print("     - stock_performance_lines.png")
print("     - allocation_breakdown.png")
