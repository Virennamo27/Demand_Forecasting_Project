# =============================================================================
# 03_arima.py  —  ARIMA Forecasting (Store 1, Weekly Sales)
# Goal: Fit an ARIMA(p,d,q) model to Store 1's aggregated weekly sales and
#       produce a 12-week-ahead forecast, then evaluate accuracy with MAPE.
#
# ARIMA recap:
#   p — number of autoregressive lags   (AR: uses past values)
#   d — degree of differencing          (I:  makes series stationary)
#   q — moving-average lags             (MA: uses past forecast errors)
#
# We use grid search over p∈[0,3], d∈[0,2], q∈[0,3] and pick the model with
# the lowest AIC (Akaike Information Criterion) on the training set.
# =============================================================================

import os
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")   # suppress convergence warnings during grid search

# ── Directories ───────────────────────────────────────────────────────────────
PLOT_DIR   = "plots/arima"
RESULT_DIR = "results"
os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

# =============================================================================
# 1. Load & Prepare Store 1 Weekly Series
# =============================================================================
train = pd.read_csv("data/train.csv", parse_dates=["Date"])
features = pd.read_csv("data/features.csv", parse_dates=["Date"])

# Aggregate all departments → store-level weekly sales for Store 1
store1 = (
    train[train["Store"] == 1]
    .groupby("Date")["Weekly_Sales"]
    .sum()
    .reset_index()
    .sort_values("Date")
)

# Merge IsHoliday flag so we can optionally remove holiday spikes
holiday_map = (
    features[features["Store"] == 1][["Date", "IsHoliday"]]
    .drop_duplicates("Date")
)
store1 = store1.merge(holiday_map, on="Date", how="left")

# Rather than dropping holiday weeks (which creates gaps that resample fills
# with 0 and destroys the MAPE), we interpolate their values so the series
# stays regular while smoothing out the extreme holiday spikes.
n_total = len(store1)
store1_full = store1.set_index("Date").sort_index()
sales_series = store1_full["Weekly_Sales"].resample("W").sum()

# Replace holiday-week values with linear interpolation of neighbours so the
# model fits the underlying non-promotional demand pattern.
holiday_mask = store1_full["IsHoliday"].reindex(sales_series.index).fillna(False)
sales_series[holiday_mask] = np.nan
sales_series = sales_series.interpolate(method="linear")

print(f"Total weeks: {n_total} | Series length after resample+interpolate: {len(sales_series)}")

print(f"\nSeries length: {len(sales_series)} weeks")
print(f"Date range: {sales_series.index.min().date()} → {sales_series.index.max().date()}")

# =============================================================================
# 2. Stationarity Check (Augmented Dickey-Fuller test)
# =============================================================================
# ARIMA requires a (weakly) stationary series.  ADF null hypothesis:
# "unit root exists" (non-stationary).  Reject H₀ at p < 0.05 → stationary.

adf_result = adfuller(sales_series, autolag="AIC")
print(f"\nADF Test  p-value: {adf_result[1]:.4f}  "
      f"{'→ stationary (p<0.05)' if adf_result[1] < 0.05 else '→ non-stationary, differencing needed'}")

# =============================================================================
# 3. ACF & PACF Plots
# =============================================================================
# ACF  (Autocorrelation Function):  guides choice of MA order (q)
# PACF (Partial Autocorrelation Function):  guides choice of AR order (p)
# We plot on both the raw and first-differenced series.

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

plot_acf(sales_series,             lags=40, ax=axes[0, 0], title="ACF — Raw Series")
plot_pacf(sales_series,            lags=40, ax=axes[0, 1], title="PACF — Raw Series",
          method="ywm")
plot_acf(sales_series.diff().dropna(),  lags=40, ax=axes[1, 0], title="ACF — 1st Difference")
plot_pacf(sales_series.diff().dropna(), lags=40, ax=axes[1, 1], title="PACF — 1st Difference",
          method="ywm")

plt.suptitle("ACF & PACF — Store 1 Weekly Sales", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_acf_pacf.png", dpi=150)
plt.close()
print("\nSaved: ACF & PACF plots")

# =============================================================================
# 4. Train / Test Split  (last 12 weeks = test)
# =============================================================================
TEST_SIZE = 12
train_series = sales_series[:-TEST_SIZE]
test_series  = sales_series[-TEST_SIZE:]

print(f"\nTrain: {len(train_series)} weeks | Test: {len(test_series)} weeks")

# =============================================================================
# 5. Grid Search — Best ARIMA(p,d,q) by AIC
# =============================================================================
p_range = range(0, 4)   # AR order candidates
d_range = range(0, 3)   # Differencing candidates
q_range = range(0, 4)   # MA order candidates

combos   = list(itertools.product(p_range, d_range, q_range))
best_aic = np.inf
best_order = None
aic_records = []

print(f"\nGrid-searching {len(combos)} ARIMA configurations …")
for order in combos:
    try:
        model = ARIMA(train_series, order=order)
        fit   = model.fit()
        aic_records.append({"order": str(order), "aic": fit.aic})
        if fit.aic < best_aic:
            best_aic   = fit.aic
            best_order = order
    except Exception:
        pass   # some parameter combos fail to converge — skip them

print(f"Best ARIMA order: {best_order}  (AIC = {best_aic:.2f})")

# =============================================================================
# 6. Fit Best ARIMA Model & Forecast
# =============================================================================
best_model = ARIMA(train_series, order=best_order)
best_fit   = best_model.fit()
print("\nModel summary:")
print(best_fit.summary())

# get_forecast returns distribution; we extract mean, 95% CI
forecast_obj = best_fit.get_forecast(steps=TEST_SIZE)
forecast_mean = forecast_obj.predicted_mean
forecast_ci   = forecast_obj.conf_int(alpha=0.05)   # 95% confidence interval

# Align forecast index to the actual test dates
forecast_mean.index = test_series.index
forecast_ci.index   = test_series.index

# =============================================================================
# 7. MAPE Score
# =============================================================================
# MAPE = mean(|actual - forecast| / |actual|) × 100
# Lower is better; <10% is generally considered a good forecast.

mape = mean_absolute_percentage_error(test_series, forecast_mean) * 100
print(f"\nARIMA MAPE: {mape:.2f}%")

# =============================================================================
# 8. Plot — Forecast vs Actuals
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 5))

# Show last 26 weeks of training for context
ax.plot(train_series.index[-26:], train_series.values[-26:] / 1e6,
        color="steelblue", linewidth=1.4, label="Training (last 26 wks)")
ax.plot(test_series.index, test_series.values / 1e6,
        color="black", linewidth=1.6, marker="o", markersize=4, label="Actual")
ax.plot(forecast_mean.index, forecast_mean.values / 1e6,
        color="darkorange", linewidth=1.6, linestyle="--",
        marker="s", markersize=4, label=f"ARIMA{best_order} Forecast")
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:, 0] / 1e6,
                forecast_ci.iloc[:, 1] / 1e6,
                alpha=0.2, color="darkorange", label="95% CI")

ax.axvline(test_series.index[0], color="gray", linestyle=":", linewidth=1.2,
           label="Train | Test split")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=30)
ax.set_title(f"ARIMA{best_order} Forecast vs Actuals — Store 1\n"
             f"MAPE = {mape:.2f}%", fontsize=13)
ax.set_ylabel("Weekly Sales ($M)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_arima_forecast.png", dpi=150)
plt.close()
print("Saved: ARIMA forecast plot")

# =============================================================================
# 9. AIC Score Plot — Top 20 Configurations
# =============================================================================
aic_df = pd.DataFrame(aic_records).sort_values("aic").head(20)

fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#e74c3c" if o == str(best_order) else "#95a5a6" for o in aic_df["order"]]
ax.barh(aic_df["order"], aic_df["aic"], color=colors)
ax.invert_yaxis()
ax.set_title("Top 20 ARIMA Configurations by AIC (lower = better)", fontsize=12)
ax.set_xlabel("AIC Score")
ax.axvline(best_aic, color="red", linestyle="--", linewidth=1, label=f"Best: {best_order}")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_aic_scores.png", dpi=150)
plt.close()
print("Saved: AIC scores plot")

# =============================================================================
# 10. Save Results CSV
# =============================================================================
results_df = pd.DataFrame({
    "Date":          test_series.index,
    "Actual_Sales":  test_series.values,
    "ARIMA_Forecast": forecast_mean.values,
    "CI_Lower":      forecast_ci.iloc[:, 0].values,
    "CI_Upper":      forecast_ci.iloc[:, 1].values,
    "MAPE":          [mape] + [np.nan] * (TEST_SIZE - 1),
    "Best_Order":    [str(best_order)] + [""] * (TEST_SIZE - 1),
})
results_df.to_csv(f"{RESULT_DIR}/arima_results.csv", index=False)
print(f"\nResults saved to: {RESULT_DIR}/arima_results.csv")

print("\n=== ARIMA Summary ===")
print(f"  Best order  : ARIMA{best_order}")
print(f"  AIC         : {best_aic:.2f}")
print(f"  MAPE        : {mape:.2f}%")
print(f"  Interpretation: {'Good (<10%)' if mape < 10 else 'Moderate (10-20%)' if mape < 20 else 'Needs improvement (>20%)'}")
