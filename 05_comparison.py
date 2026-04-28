# =============================================================================
# 05_comparison.py  —  Model Comparison: ARIMA vs Prophet
# Goal: Load the saved forecast results from both models, compare accuracy
#       metrics, visualise forecasts side-by-side, and summarise findings.
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# ── Directories ───────────────────────────────────────────────────────────────
RESULT_DIR = "results"
PLOT_DIR   = "plots"          # comparison plots go into top-level plots/ dir
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

# =============================================================================
# 1. Load Results
# =============================================================================
arima_df   = pd.read_csv(f"{RESULT_DIR}/arima_results.csv",   parse_dates=["Date"])
prophet_df = pd.read_csv(f"{RESULT_DIR}/prophet_results.csv", parse_dates=["Date"])

# Extract MAPE (stored only in the first row)
arima_mape   = arima_df["MAPE"].dropna().iloc[0]
prophet_mape = prophet_df["MAPE"].dropna().iloc[0]

print(f"ARIMA   MAPE: {arima_mape:.2f}%")
print(f"Prophet MAPE: {prophet_mape:.2f}%")

# Determine best model
best_model = "ARIMA" if arima_mape < prophet_mape else "Prophet"
mape_diff  = abs(arima_mape - prophet_mape)
print(f"Best model  : {best_model} (lower MAPE by {mape_diff:.2f} percentage points)")

# =============================================================================
# 2. MAPE Comparison Bar Chart
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 5))

models = ["ARIMA", "Prophet"]
mapes  = [arima_mape, prophet_mape]
colors = ["#2196F3", "#4CAF50"]
bar_highlight = ["#e74c3c" if m == best_model else c
                 for m, c in zip(models, colors)]

bars = ax.bar(models, mapes, color=bar_highlight, width=0.4, edgecolor="white",
              linewidth=1.5)
ax.set_title("Model Comparison — MAPE (lower is better)", fontsize=13)
ax.set_ylabel("MAPE (%)")
ax.set_ylim(0, max(mapes) * 1.35)

for bar, val in zip(bars, mapes):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{val:.2f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

# Add a star annotation on the best model
best_idx = models.index(best_model)
ax.text(best_idx, mapes[best_idx] / 2, "* Best",
        ha="center", va="center", fontsize=11, color="white", fontweight="bold")

ax.axhline(10, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.text(1.45, 10.1, "10% threshold", fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_mape_comparison.png", dpi=150)
plt.close()
print("\nSaved: MAPE comparison bar chart")

# =============================================================================
# 3. Forecast vs Actuals Overlay — Side by Side
# =============================================================================
fig = plt.figure(figsize=(16, 6))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

# ── ARIMA panel ──────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(arima_df["Date"], arima_df["Actual_Sales"] / 1e6,
         color="black", linewidth=1.8, marker="o", markersize=5, label="Actual")
ax1.plot(arima_df["Date"], arima_df["ARIMA_Forecast"] / 1e6,
         color="#2196F3", linewidth=1.6, linestyle="--",
         marker="s", markersize=4, label="ARIMA Forecast")
ax1.fill_between(arima_df["Date"],
                 arima_df["CI_Lower"] / 1e6,
                 arima_df["CI_Upper"] / 1e6,
                 alpha=0.15, color="#2196F3", label="95% CI")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax1.set_title(f"ARIMA  —  MAPE: {arima_mape:.2f}%", fontsize=12)
ax1.set_ylabel("Weekly Sales ($M)")
ax1.legend(fontsize=9)

# ── Prophet panel ─────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.plot(prophet_df["Date"], prophet_df["Actual_Sales"] / 1e6,
         color="black", linewidth=1.8, marker="o", markersize=5, label="Actual")
ax2.plot(prophet_df["Date"], prophet_df["Prophet_Forecast"] / 1e6,
         color="#4CAF50", linewidth=1.6, linestyle="--",
         marker="s", markersize=4, label="Prophet Forecast")
ax2.fill_between(prophet_df["Date"],
                 prophet_df["CI_Lower"] / 1e6,
                 prophet_df["CI_Upper"] / 1e6,
                 alpha=0.15, color="#4CAF50", label="95% CI")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax2.set_title(f"Prophet  —  MAPE: {prophet_mape:.2f}%", fontsize=12)
ax2.set_ylabel("Weekly Sales ($M)")
ax2.legend(fontsize=9)

plt.suptitle("Forecast vs Actuals — ARIMA vs Prophet (Store 1, Last 12 Weeks)",
             fontsize=13, y=1.02)
plt.savefig(f"{PLOT_DIR}/02_forecast_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: forecast overlay chart")

# =============================================================================
# 4. Combined Overlay — Both Forecasts on One Chart
# =============================================================================
fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(arima_df["Date"], arima_df["Actual_Sales"] / 1e6,
        color="black", linewidth=2, marker="o", markersize=5, zorder=5, label="Actual")
ax.plot(arima_df["Date"], arima_df["ARIMA_Forecast"] / 1e6,
        color="#2196F3", linewidth=1.6, linestyle="--", marker="s", markersize=4,
        label=f"ARIMA (MAPE {arima_mape:.1f}%)")
ax.plot(prophet_df["Date"], prophet_df["Prophet_Forecast"] / 1e6,
        color="#4CAF50", linewidth=1.6, linestyle="-.", marker="^", markersize=4,
        label=f"Prophet (MAPE {prophet_mape:.1f}%)")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=30)
ax.set_title("ARIMA vs Prophet — Forecast Comparison (Store 1)", fontsize=13)
ax.set_ylabel("Weekly Sales ($M)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_combined_overlay.png", dpi=150)
plt.close()
print("Saved: combined overlay chart")

# =============================================================================
# 5. Error Distribution Comparison
# =============================================================================
arima_errors   = ((arima_df["Actual_Sales"]   - arima_df["ARIMA_Forecast"])
                  / arima_df["Actual_Sales"]   * 100)
prophet_errors = ((prophet_df["Actual_Sales"] - prophet_df["Prophet_Forecast"])
                  / prophet_df["Actual_Sales"] * 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

axes[0].bar(range(1, len(arima_errors) + 1), arima_errors,
            color=["#e74c3c" if e > 0 else "#2196F3" for e in arima_errors])
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title(f"ARIMA Percentage Errors per Week\n(Mean: {arima_errors.mean():.1f}%)", fontsize=11)
axes[0].set_xlabel("Week (Test Period)")
axes[0].set_ylabel("% Error")

axes[1].bar(range(1, len(prophet_errors) + 1), prophet_errors,
            color=["#e74c3c" if e > 0 else "#4CAF50" for e in prophet_errors])
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title(f"Prophet Percentage Errors per Week\n(Mean: {prophet_errors.mean():.1f}%)", fontsize=11)
axes[1].set_xlabel("Week (Test Period)")

plt.suptitle("Forecast Error Distribution — Test Period", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_error_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: error distribution chart")

# =============================================================================
# 6. Export Comparison Results CSV
# =============================================================================
comparison_df = pd.DataFrame({
    "Date":             arima_df["Date"],
    "Actual_Sales":     arima_df["Actual_Sales"],
    "ARIMA_Forecast":   arima_df["ARIMA_Forecast"],
    "Prophet_Forecast": prophet_df["Prophet_Forecast"],
    "ARIMA_Pct_Error":  arima_errors.values,
    "Prophet_Pct_Error": prophet_errors.values,
})
comparison_df.to_csv(f"{RESULT_DIR}/comparison_results.csv", index=False)
print(f"\nComparison results saved to: {RESULT_DIR}/comparison_results.csv")

# =============================================================================
# 7. Final Summary Printout
# =============================================================================
better_by = abs(arima_mape - prophet_mape)

print("\n" + "=" * 55)
print("  FINAL MODEL COMPARISON SUMMARY")
print("=" * 55)
print(f"  ARIMA MAPE         :  {arima_mape:.2f}%")
print(f"  Prophet MAPE       :  {prophet_mape:.2f}%")
print(f"  Best Model         :  {best_model}  (by {better_by:.2f} pp)")
print("-" * 55)

# Business insight based on which model won
if best_model == "Prophet":
    insight = (
        "Prophet's advantage likely comes from its explicit holiday "
        "components and the Temperature / Fuel_Price regressors, which "
        "capture demand drivers that a pure time-series ARIMA cannot "
        "see.  For planning purposes, incorporating macro and event data "
        "into the forecast is worth the added complexity."
    )
else:
    insight = (
        "ARIMA's edge suggests the sales pattern in this window is "
        "dominated by its own lag structure — past sales are the strongest "
        "predictor of near-future sales — and the additional holiday / "
        "regressor signals in Prophet may be introducing noise rather than "
        "signal for this particular test period.  ARIMA is simpler and "
        "quicker to re-train, making it a strong operational baseline."
    )

# Wrap insight text for clean terminal display
import textwrap
wrapped = textwrap.fill(insight, width=55, initial_indent="  ", subsequent_indent="  ")
print(f"  Key Business Insight:")
print(wrapped)
print("=" * 55)
print(f"\nAll comparison plots saved to: {PLOT_DIR}/")
