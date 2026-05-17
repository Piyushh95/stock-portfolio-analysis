import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("\n" + "=" * 80)
print("MACHINE LEARNING: Predicting Stock Returns")
print("=" * 80)

# Load data
returns_df = pd.read_csv("../analysis/returns_timeseries.csv", index_col="Date", parse_dates=True)

# ===== FEATURE ENGINEERING =====
print("\nFEATURE ENGINEERING")
print("-" * 80)


def create_features(returns_series, lags=(1, 5, 20), name="Stock"):
    """Create lagged features and rolling statistics."""
    df = returns_series.copy().to_frame(name=f"{name}_return")

    for lag in lags:
        df[f"{name}_lag_{lag}"] = returns_series.shift(lag)

    df[f"{name}_vol_20"] = returns_series.rolling(20).std()
    df[f"{name}_mom_5"] = returns_series.rolling(5).mean()
    df[f"{name}_ema_10"] = returns_series.ewm(span=10).mean()
    df[f"{name}_next_return"] = returns_series.shift(-1)
    return df.dropna()


top_stock = "GOOGL" if "GOOGL" in returns_df.columns else returns_df.columns[0]
features = create_features(returns_df[top_stock], name=top_stock)
print(f"Created features for {top_stock}")
print(f"Shape: {features.shape}")

# ===== PREPARE DATA FOR MODELING =====
print("\nPREPARING DATA")
print("-" * 80)

target_col = f"{top_stock}_next_return"
feature_cols = [col for col in features.columns if col not in (f"{top_stock}_return", target_col)]

X = features[feature_cols].values
y = features[target_col].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Features: {len(feature_cols)}")

# ===== MODEL TRAINING =====
print("\nTRAINING MODELS")
print("-" * 80)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest (100 trees)": RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=10
    ),
    "Random Forest (50 trees)": RandomForestRegressor(
        n_estimators=50, random_state=42, max_depth=5
    ),
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    results[model_name] = {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "model": model,
        "y_test_pred": y_test_pred,
    }

    print(f"\n{model_name}")
    print(f"  Train R^2: {train_r2:.4f} | Test R^2: {test_r2:.4f}")
    print(f"  Train RMSE: {train_rmse:.6f} | Test RMSE: {test_rmse:.6f}")
    print(f"  Train MAE: {train_mae:.6f} | Test MAE: {test_mae:.6f}")

# ===== FEATURE IMPORTANCE =====
print("\nFEATURE IMPORTANCE (Random Forest)")
print("-" * 80)

rf_model = results["Random Forest (100 trees)"]["model"]
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values(
    "Importance", ascending=False
)

print("\nTop 5 Most Important Features:")
for _, row in importance_df.head().iterrows():
    print(f"  {row['Feature']:20s}: {row['Importance']:.4f}")

# ===== VISUALIZATION =====
print("\nCREATING VISUALIZATIONS")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
lr_preds = results["Linear Regression"]["y_test_pred"]
ax.scatter(y_test, lr_preds, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
ax.set_xlabel("Actual Return")
ax.set_ylabel("Predicted Return")
ax.set_title(f'Linear Regression (R^2 = {results["Linear Regression"]["test_r2"]:.3f})')
ax.grid(alpha=0.3)

ax = axes[0, 1]
rf_preds = results["Random Forest (100 trees)"]["y_test_pred"]
ax.scatter(y_test, rf_preds, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
ax.set_xlabel("Actual Return")
ax.set_ylabel("Predicted Return")
ax.set_title(f'Random Forest (R^2 = {results["Random Forest (100 trees)"]["test_r2"]:.3f})')
ax.grid(alpha=0.3)

ax = axes[1, 0]
importance_df.head(10).plot(x="Feature", y="Importance", kind="barh", ax=ax)
ax.set_xlabel("Importance")
ax.set_title("Top 10 Feature Importance")

ax = axes[1, 1]
model_names = list(results.keys())
test_r2_scores = [results[m]["test_r2"] for m in model_names]
test_rmse_scores = [results[m]["test_rmse"] for m in model_names]
x = np.arange(len(model_names))
ax2 = ax.twinx()
ax.bar(x - 0.2, test_r2_scores, 0.4, label="R^2 Score", color="skyblue")
ax2.bar(x + 0.2, test_rmse_scores, 0.4, label="RMSE", color="orange")
ax.set_ylabel("R^2 Score", color="skyblue")
ax2.set_ylabel("RMSE", color="orange")
ax.set_xlabel("Model")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels([m.replace(" (", "\n(") for m in model_names], fontsize=9)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("../analysis/ml_analysis.png", dpi=300, bbox_inches="tight")
print("Saved: analysis/ml_analysis.png")

print("\n" + "=" * 80)
print("ML Analysis Complete")
print("=" * 80)

# ===== KEY INSIGHTS =====
print("\nKEY INSIGHTS")
print("-" * 80)
best_model = max(results.items(), key=lambda x: x[1]["test_r2"])
print(f"Best Model: {best_model[0]}")
print(f"Test R^2: {best_model[1]['test_r2']:.4f}")
print(f"Test RMSE: {best_model[1]['test_rmse']:.6f}")
