# =============================================================================
# 04_prophet.py  —  Facebook/Meta Prophet Forecasting (Store 1, Weekly Sales)
# Goal: Fit a Prophet model with holidays, regressors, and Walmart-specific
#       events to Store 1's aggregated weekly sales, then forecast 12 weeks.
#
# Why Prophet?
#   • Handles multiple seasonalities (weekly, yearly) out of the box
#   • Has explicit holiday / special-event components
#   • Accepts external regressors (Temperature, Fuel_Price)
#   • Robust to missing data and outliers
#   • Decomposable — we can inspect every additive component separately
# =============================================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from prophet.plot import plot_components_plotly
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# ── Directories ───────────────────────────────────────────────────────────────
PLOT_DIR   = "plots/prophet"
RESULT_DIR = "results"
os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

# =============================================================================
# 1. Load & Prepare Data
# =============================================================================
train    = pd.read_csv("data/train.csv",    parse_dates=["Date"])
features = pd.read_csv("data/features.csv", parse_dates=["Date"])

# Aggregate all departments in Store 1
store1_sales = (
    train[train["Store"] == 1]
    .groupby("Date")["Weekly_Sales"]
    .sum()
    .reset_index()
    .rename(columns={"Date": "ds", "Weekly_Sales": "y"})
    .sort_values("ds")
)

# Pull macro regressors for Store 1 — Temperature and Fuel_Price
store1_features = (
    features[features["Store"] == 1][["Date", "Temperature", "Fuel_Price", "IsHoliday"]]
    .rename(columns={"Date": "ds"})
    .drop_duplicates("ds")
)

# Merge regressors into the main dataframe
df = store1_sales.merge(store1_features, on="ds", how="left")

# Forward-fill any minor gaps in regressor values
df["Temperature"] = df["Temperature"].ffill()
df["Fuel_Price"]  = df["Fuel_Price"].ffill()

print(f"Prophet dataframe shape: {df.shape}")
print(df.head(3).to_string())

# =============================================================================
# 2. Build Holiday DataFrames
# =============================================================================
# Prophet accepts holidays as a DataFrame with columns: holiday, ds, lower_window,
# upper_window.  We include:
#   (a) US Federal holidays — via the holidays library (bundled with Prophet)
#   (b) Walmart's four "super-event" holidays (as defined in the dataset)

# ── (a) US Federal Holidays via Prophet's built-in country_holidays ───────────
# These are specified at model init with country_holidays="US"

# ── (b) Walmart-specific events ──────────────────────────────────────────────
# The dataset's IsHoliday flag covers: Super Bowl (Feb), Labor Day (Sep),
# Thanksgiving (Nov), Christmas (Dec).
# We build a proper Prophet holiday table for these four events covering
# every year present in the data.

years = df["ds"].dt.year.unique()

walmart_events = []
for year in sorted(years):
    # Super Bowl — first Sunday of February
    feb1  = pd.Timestamp(year, 2, 1)
    super_bowl = feb1 + pd.offsets.Week(weekday=6) if feb1.weekday() != 6 else feb1
    walmart_events.append({"holiday": "SuperBowl",    "ds": super_bowl,
                            "lower_window": 0, "upper_window": 1})

    # Labor Day — first Monday of September
    sep1  = pd.Timestamp(year, 9, 1)
    days_to_mon = (7 - sep1.weekday()) % 7
    labor_day   = sep1 + pd.Timedelta(days=days_to_mon)
    walmart_events.append({"holiday": "LaborDay",     "ds": labor_day,
                            "lower_window": 0, "upper_window": 1})

    # Thanksgiving — 4th Thursday of November
    nov1   = pd.Timestamp(year, 11, 1)
    days_to_thu = (3 - nov1.weekday()) % 7   # 3 = Thursday
    first_thu   = nov1 + pd.Timedelta(days=days_to_thu)
    thanksgiving = first_thu + pd.Timedelta(weeks=3)
    walmart_events.append({"holiday": "Thanksgiving", "ds": thanksgiving,
                            "lower_window": 0, "upper_window": 1})

    # Christmas — Dec 25 (effect felt the week before)
    walmart_events.append({"holiday": "Christmas",    "ds": pd.Timestamp(year, 12, 25),
                            "lower_window": -1, "upper_window": 1})

holidays_df = pd.DataFrame(walmart_events)
print(f"\nWalmart holiday events built: {len(holidays_df)} rows")

# =============================================================================
# 3. Train / Test Split  (last 12 weeks = test)
# =============================================================================
TEST_SIZE  = 12
train_df   = df.iloc[:-TEST_SIZE].copy()
test_df    = df.iloc[-TEST_SIZE:].copy()

print(f"\nTrain: {len(train_df)} weeks | Test: {len(test_df)} weeks")

# =============================================================================
# 4. Fit Prophet Model
# =============================================================================
model = Prophet(
    yearly_seasonality=True,     # captures annual pattern (52-week cycle)
    weekly_seasonality=False,    # no intra-week variation (we have one obs/week)
    daily_seasonality=False,
    holidays=holidays_df,        # Walmart events
    seasonality_mode="additive", # additive is appropriate when variance is stable
    interval_width=0.95,         # 95% uncertainty interval
)
# add_country_holidays is the correct API in Prophet >= 1.0
model.add_country_holidays(country_name="US")

# Add external regressors — Prophet treats these as additional linear components
# Each regressor must exist in both train and future DataFrames
model.add_regressor("Temperature", standardize=True)
model.add_regressor("Fuel_Price",  standardize=True)

model.fit(train_df[["ds", "y", "Temperature", "Fuel_Price"]])
print("Prophet model fitted successfully.")

# =============================================================================
# 5. Forecast
# =============================================================================
# Build a future DataFrame that includes the test period's known regressors.
# Prophet's make_future_dataframe covers the full history + forecast horizon.

# freq="W-FRI" matches Walmart's Friday-ending weekly cadence
future = model.make_future_dataframe(periods=TEST_SIZE, freq="W-FRI")

# Attach regressor values for the future period
future = future.merge(
    df[["ds", "Temperature", "Fuel_Price"]],
    on="ds", how="left"
)
# For weeks beyond the dataset we forward-fill from the last known value
future["Temperature"] = future["Temperature"].ffill()
future["Fuel_Price"]  = future["Fuel_Price"].ffill()

forecast = model.predict(future)

# Separate test-period predictions — use the last TEST_SIZE rows of the
# forecast (which correspond to the forecast horizon beyond training history)
forecast_test = forecast.tail(TEST_SIZE).copy().reset_index(drop=True)
test_df       = test_df.reset_index(drop=True)

# =============================================================================
# 6. MAPE Score
# =============================================================================
mape = mean_absolute_percentage_error(test_df["y"], forecast_test["yhat"]) * 100
print(f"\nProphet MAPE: {mape:.2f}%")

# =============================================================================
# 7. Plot — Forecast vs Actuals
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 5))

# Show last 26 training weeks for context
train_context = train_df.set_index("ds").iloc[-26:]
ax.plot(train_context.index, train_context["y"] / 1e6,
        color="steelblue", linewidth=1.4, label="Training (last 26 wks)")
ax.plot(forecast_test["ds"], test_df["y"] / 1e6,
        color="black", linewidth=1.6, marker="o", markersize=4, label="Actual")
ax.plot(forecast_test["ds"], forecast_test["yhat"] / 1e6,
        color="mediumseagreen", linewidth=1.6, linestyle="--",
        marker="s", markersize=4, label="Prophet Forecast")
ax.fill_between(forecast_test["ds"],
                forecast_test["yhat_lower"] / 1e6,
                forecast_test["yhat_upper"] / 1e6,
                alpha=0.2, color="mediumseagreen", label="95% Uncertainty Interval")

ax.axvline(forecast_test["ds"].iloc[0], color="gray", linestyle=":", linewidth=1.2,
           label="Train | Test split")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=30)
ax.set_title(f"Prophet Forecast vs Actuals — Store 1\nMAPE = {mape:.2f}%", fontsize=13)
ax.set_ylabel("Weekly Sales ($M)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_prophet_forecast.png", dpi=150)
plt.close()
print("Saved: Prophet forecast plot")

# =============================================================================
# 8. Plot — Prophet Components (Trend + Seasonality + Holidays + Regressors)
# =============================================================================
# Prophet's built-in component plot shows exactly how each piece contributes.
fig_comp = model.plot_components(forecast)
fig_comp.suptitle("Prophet Components — Store 1 Weekly Sales", fontsize=13, y=1.01)
plt.tight_layout()
fig_comp.savefig(f"{PLOT_DIR}/02_prophet_components.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: Prophet components plot")

# =============================================================================
# 9. Plot — Full Prophet Forecast with History
# =============================================================================
fig_full = model.plot(forecast, figsize=(14, 5))
fig_full.axes[0].set_title("Prophet Full Forecast — Store 1 Weekly Sales", fontsize=13)
fig_full.axes[0].set_ylabel("Weekly Sales ($)")
plt.tight_layout()
fig_full.savefig(f"{PLOT_DIR}/03_prophet_full_forecast.png", dpi=150)
plt.close()
print("Saved: Prophet full forecast plot")

# =============================================================================
# 10. Holiday Impact Analysis
# =============================================================================
# Extract the magnitude of each holiday's effect from the forecast DataFrame
holiday_cols = [c for c in forecast.columns if c.startswith("holidays")
                or c in ["SuperBowl", "LaborDay", "Thanksgiving", "Christmas"]]

# Prophet stores holiday effects in a combined "holidays" column
if "holidays" in forecast.columns:
    holiday_effect = forecast[["ds", "holidays"]].dropna()
    holiday_effect = holiday_effect[holiday_effect["holidays"].abs() > 100]  # filter noise

    if not holiday_effect.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(holiday_effect["ds"], holiday_effect["holidays"] / 1e6,
               color=["#e74c3c" if v > 0 else "#3498db"
                      for v in holiday_effect["holidays"]])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.xticks(rotation=30)
        ax.set_title("Holiday Effects Estimated by Prophet", fontsize=12)
        ax.set_ylabel("Holiday Effect ($M)")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/04_holiday_effects.png", dpi=150)
        plt.close()
        print("Saved: Holiday effects plot")

# =============================================================================
# 11. Save Results CSV
# =============================================================================
results_df = pd.DataFrame({
    "Date":             forecast_test["ds"].values,
    "Actual_Sales":     test_df["y"].values,
    "Prophet_Forecast": forecast_test["yhat"].values,
    "CI_Lower":         forecast_test["yhat_lower"].values,
    "CI_Upper":         forecast_test["yhat_upper"].values,
    "MAPE":             [mape] + [np.nan] * (TEST_SIZE - 1),
})
results_df.to_csv(f"{RESULT_DIR}/prophet_results.csv", index=False)
print(f"\nResults saved to: {RESULT_DIR}/prophet_results.csv")

# =============================================================================
# 12. Regressor Coefficient Summary
# =============================================================================
print("\n=== Regressor Coefficients (standardised) ===")
for name in ["Temperature", "Fuel_Price"]:
    col = f"extra_regressors_additive" if f"{name}" not in forecast.columns else name
print("  (See Prophet components plot for regressor trend impact)")

print("\n=== Prophet Summary ===")
print(f"  MAPE        : {mape:.2f}%")
print(f"  Interpretation: {'Good (<10%)' if mape < 10 else 'Moderate (10-20%)' if mape < 20 else 'Needs improvement (>20%)'}")
print(f"\nAll Prophet plots saved to: {PLOT_DIR}/")
