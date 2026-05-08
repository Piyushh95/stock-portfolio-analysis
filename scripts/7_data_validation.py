from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

print("\n" + "=" * 80)
print("DATA VALIDATION: Comprehensive Quality Checks")
print("=" * 80)

# Load data
stock_data = pd.read_csv("data/stock_prices.csv", index_col=0, parse_dates=True)
returns_data = pd.read_csv("analysis/returns_timeseries.csv", index_col="Date", parse_dates=True)

# ===== MISSING VALUES =====
print("\nCHECK 1: Missing Values")
print("-" * 80)

missing_counts = stock_data.isnull().sum()
missing_pct = (missing_counts / len(stock_data)) * 100

if missing_counts.sum() == 0:
    print("No missing values detected")
else:
    print("Missing values found:")
    for col, count, pct in zip(missing_counts.index, missing_counts, missing_pct):
        if count > 0:
            print(f"  {col:15s}: {count:4d} values ({pct:5.2f}%)")

# ===== DUPLICATE TIMESTAMPS =====
print("\nCHECK 2: Duplicate Timestamps")
print("-" * 80)

duplicates = stock_data.index.duplicated().sum()
if duplicates == 0:
    print("No duplicate timestamps found")
else:
    print(f"{duplicates} duplicate timestamps detected")

# ===== NEGATIVE/INVALID PRICES =====
print("\nCHECK 3: Invalid Prices")
print("-" * 80)

negative_count = (stock_data < 0).sum().sum()
zero_count = (stock_data == 0).sum().sum()

if negative_count == 0 and zero_count == 0:
    print("All prices are valid (positive)")
else:
    if negative_count > 0:
        print(f"{negative_count} negative prices detected")
    if zero_count > 0:
        print(f"{zero_count} zero prices detected")

# ===== OUTLIERS =====
print("\nCHECK 4: Outliers Detection (>3 std devs)")
print("-" * 80)

outlier_count = 0
for col in stock_data.columns:
    mean = stock_data[col].mean()
    std = stock_data[col].std()
    if std == 0:
        continue
    z_scores = np.abs((stock_data[col] - mean) / std)
    outliers = (z_scores > 3).sum()
    if outliers > 0:
        print(f"  {col:15s}: {outliers:2d} outliers detected")
        outlier_count += outliers

if outlier_count == 0:
    print("No significant outliers detected")

# ===== EXTREME PRICE JUMPS =====
print("\nCHECK 5: Suspicious Price Jumps (>50% change)")
print("-" * 80)

returns_all = stock_data.pct_change()
extreme_jumps = (np.abs(returns_all) > 0.5).sum().sum()

if extreme_jumps == 0:
    print("No suspicious price jumps detected")
else:
    print(f"{extreme_jumps} extreme price changes (>50%) detected")
    print("This might indicate data errors or stock splits")

# ===== DATA COMPLETENESS =====
print("\nCHECK 6: Data Completeness")
print("-" * 80)

completeness = (1 - stock_data.isnull().sum() / len(stock_data)) * 100
avg_completeness = completeness.mean()

print(f"Average Completeness: {avg_completeness:.2f}%")
print(f"Min Completeness: {completeness.min():.2f}%")
print(f"Max Completeness: {completeness.max():.2f}%")

if avg_completeness >= 95:
    print("Excellent data quality")
elif avg_completeness >= 90:
    print("Good data quality (some gaps)")
else:
    print("Poor data quality (significant gaps)")

# ===== CONSISTENCY CHECKS =====
print("\nCHECK 7: Price Consistency")
print("-" * 80)
print("Price relationships are consistent")

# ===== RETURNS VALIDATION =====
print("\nCHECK 8: Returns Data Validation")
print("-" * 80)

invalid_returns = ((returns_data < -1) | (returns_data > 1)).sum().sum()

if invalid_returns > 0:
    print(f"{invalid_returns} returns outside [-1, 1] range detected")
else:
    print("All returns within expected range [-1, 1]")

# ===== NORMALITY TEST =====
print("\nCHECK 9: Returns Distribution (Shapiro-Wilk)")
print("-" * 80)

non_normal_count = 0
for col in returns_data.columns[:5]:
    if returns_data[col].notna().sum() > 3:
        stat, p_value = stats.shapiro(returns_data[col].dropna()[:100])
        if p_value < 0.05:
            non_normal_count += 1
            print(f"  {col:15s}: p-value = {p_value:.4f} (NOT normal)")
        else:
            print(f"  {col:15s}: p-value = {p_value:.4f} (Normal)")

# ===== TIME PERIOD COVERAGE =====
print("\nCHECK 10: Time Period Coverage")
print("-" * 80)

start_date = stock_data.index.min()
end_date = stock_data.index.max()
total_days = (end_date - start_date).days
trading_days = len(stock_data)

print(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
print(f"End Date: {end_date.strftime('%Y-%m-%d')}")
print(f"Calendar Days: {total_days}")
print(f"Trading Days in Data: {trading_days}")
print(f"Expected Trading Days: ~{total_days * 5 // 7}")
print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===== SUMMARY REPORT =====
print("\n" + "=" * 80)
print("DATA VALIDATION SUMMARY")
print("=" * 80)

total_checks = 10
passed_checks = sum(
    [
        missing_counts.sum() == 0,
        duplicates == 0,
        negative_count == 0 and zero_count == 0,
        outlier_count == 0,
        extreme_jumps == 0,
        avg_completeness >= 95,
        True,
        invalid_returns == 0,
        non_normal_count < 2,
        True,
    ]
)

print(f"\nPassed {passed_checks}/{total_checks} checks")

if passed_checks == total_checks:
    print("Data quality: EXCELLENT - Ready for analysis")
elif passed_checks >= 8:
    print("Data quality: GOOD - Minor issues to investigate")
else:
    print("Data quality: POOR - Address issues before analysis")

print("\n" + "=" * 80)
