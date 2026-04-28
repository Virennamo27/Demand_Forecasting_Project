# =============================================================================
# 02_stl_decomposition.py  —  STL (Seasonal-Trend decomposition using LOESS)
# Goal: Break the Store 1 weekly sales time series into its three fundamental
#       components — Trend, Seasonality, and Residual — so we can understand
#       the underlying structure before fitting statistical models.
#
# Why STL?
#   • Robust to outliers (LOESS smoother is less sensitive than classical X11)
#   • Handles any periodicity (we use period=52 for annual weekly seasonality)
#   • Decomposition is additive: Observed = Trend + Seasonal + Residual
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL

# ── Output directory ──────────────────────────────────────────────────────────
PLOT_DIR = "plots/stl"
os.makedirs(PLOT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

# =============================================================================
# 1. Load & Filter to Store 1
# =============================================================================
train = pd.read_csv("data/train.csv", parse_dates=["Date"])

# Aggregate all departments within Store 1 into a single weekly sales figure.
# This gives a clean store-level signal suitable for univariate decomposition.
store1 = (
    train[train["Store"] == 1]
    .groupby("Date")["Weekly_Sales"]
    .sum()
    .reset_index()
    .sort_values("Date")
)

print(f"Store 1 series: {len(store1)} weekly observations")
print(f"Date range: {store1['Date'].min().date()} → {store1['Date'].max().date()}")

# Resample to ensure perfectly regular weekly frequency (no gaps).
# Setting the index to DatetimeIndex and using resample fills implicit gaps.
store1 = store1.set_index("Date")
store1 = store1["Weekly_Sales"].resample("W").sum()

print(f"\nAfter resample: {len(store1)} weeks")
print(f"Any NaN after resample: {store1.isnull().sum()}")

# =============================================================================
# 2. Apply STL Decomposition
# =============================================================================
# period=52  → one full seasonal cycle = 52 weeks (annual)
# seasonal=13 → window length for the LOESS seasonal smoother (odd number,
#               roughly 13 weeks gives smooth but responsive seasonal curve)
# robust=True → downweights outlier residuals so spikes don't distort trend

stl    = STL(store1, period=52, seasonal=13, robust=True)
result = stl.fit()

# Extract the three components
trend    = result.trend
seasonal = result.seasonal
residual = result.resid

# Strength metrics (Wang et al. 2006) — how much variance each component
# explains relative to the de-seasonalised / de-trended series
var_resid  = np.var(residual)
Ft = max(0, 1 - var_resid / np.var(trend + residual))    # Trend strength
Fs = max(0, 1 - var_resid / np.var(seasonal + residual)) # Seasonal strength

print(f"\nTrend strength  (Ft): {Ft:.3f}  (0 = no trend, 1 = pure trend)")
print(f"Seasonal strength (Fs): {Fs:.3f}  (0 = no seasonality, 1 = pure seasonal)")

# =============================================================================
# 3. Plot — Full Decomposition (4-panel)
# =============================================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

date_index = store1.index

axes[0].plot(date_index, store1.values / 1e6,      color="steelblue",  linewidth=1.2)
axes[0].set_title("Observed", fontsize=11)
axes[0].set_ylabel("Sales ($M)")

axes[1].plot(date_index, trend.values / 1e6,        color="darkorange", linewidth=1.5)
axes[1].set_title(f"Trend  (strength = {Ft:.2f})", fontsize=11)
axes[1].set_ylabel("Sales ($M)")

axes[2].plot(date_index, seasonal.values / 1e6,     color="green",      linewidth=1.2)
axes[2].axhline(0, color="black", linewidth=0.6, linestyle="--")
axes[2].set_title(f"Seasonal  (strength = {Fs:.2f})", fontsize=11)
axes[2].set_ylabel("Seasonal ($M)")

axes[3].plot(date_index, residual.values / 1e6,     color="crimson",    linewidth=1.0, alpha=0.8)
axes[3].axhline(0, color="black", linewidth=0.6, linestyle="--")
axes[3].set_title("Residual", fontsize=11)
axes[3].set_ylabel("Residual ($M)")
axes[3].set_xlabel("Date")

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.suptitle("STL Decomposition — Store 1 Weekly Sales\n"
             "(Trend + Seasonal + Residual)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_stl_full_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: full STL decomposition plot")

# =============================================================================
# 4. Plot — Seasonal Pattern (average week profile)
# =============================================================================
# Average each week-of-year's seasonal component across all years to produce
# a "typical year" profile — reveals which weeks naturally spike or dip.

seasonal_df = pd.DataFrame({"Date": date_index, "Seasonal": seasonal.values})
seasonal_df["Week"] = seasonal_df["Date"].dt.isocalendar().week.astype(int)
avg_seasonal = seasonal_df.groupby("Week")["Seasonal"].mean()

fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(avg_seasonal.index, avg_seasonal.values / 1e6,
       color=["#e74c3c" if v > 0 else "#3498db" for v in avg_seasonal.values],
       width=0.8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Average Seasonal Component by Week-of-Year (Store 1)", fontsize=12)
ax.set_xlabel("Week of Year")
ax.set_ylabel("Seasonal Effect ($M)")
ax.set_xticks(range(1, 53, 4))
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_seasonal_profile.png", dpi=150)
plt.close()
print("Saved: seasonal week profile")

# =============================================================================
# 5. Plot — Trend with Confidence Band (±1 std of residual)
# =============================================================================
residual_std = residual.std()

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(date_index,
                (trend - residual_std) / 1e6,
                (trend + residual_std) / 1e6,
                alpha=0.25, color="darkorange", label="±1σ residual band")
ax.plot(date_index, trend / 1e6, color="darkorange", linewidth=2, label="Trend")
ax.plot(date_index, store1 / 1e6, color="steelblue", linewidth=0.8,
        alpha=0.6, label="Observed")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=30)
ax.set_title("Trend Component with Residual Uncertainty Band", fontsize=12)
ax.set_ylabel("Sales ($M)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_trend_with_band.png", dpi=150)
plt.close()
print("Saved: trend with confidence band")

# =============================================================================
# 6. Key Findings Printout
# =============================================================================
# Identify the peak and trough weeks in the seasonal component
peak_week  = avg_seasonal.idxmax()
trough_week = avg_seasonal.idxmin()

trend_start = trend.iloc[4]    # skip first few (STL edge artefacts)
trend_end   = trend.iloc[-5]
trend_direction = "upward" if trend_end > trend_start else "downward"
trend_pct = (trend_end - trend_start) / trend_start * 100

print("\n=== STL Decomposition — Key Findings ===")
print(f"  Trend direction : {trend_direction} ({trend_pct:+.1f}% over the full period)")
print(f"  Trend strength  : {Ft:.2f} → ", end="")
print("strong trend" if Ft > 0.6 else ("moderate trend" if Ft > 0.3 else "weak trend"))
print(f"  Seasonal strength: {Fs:.2f} → ", end="")
print("strong seasonality" if Fs > 0.6 else ("moderate" if Fs > 0.3 else "weak"))
print(f"  Peak season week : Week {peak_week} "
      f"(~{avg_seasonal[peak_week]/1e6:+.2f}M above trend)")
print(f"  Trough season wk : Week {trough_week} "
      f"(~{avg_seasonal[trough_week]/1e6:+.2f}M below trend)")
print(f"  Residual std     : ${residual_std:,.0f} per week")
print(f"\nAll STL plots saved to: {PLOT_DIR}/")
