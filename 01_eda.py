# =============================================================================
# 01_eda.py  —  Exploratory Data Analysis
# Goal: Understand the structure, distributions, and key patterns in the
#       Walmart Store Sales dataset before building any forecasting models.
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ── Output directory ──────────────────────────────────────────────────────────
PLOT_DIR = "plots/eda"
os.makedirs(PLOT_DIR, exist_ok=True)

# Use a clean, presentation-ready style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")

# =============================================================================
# 1. Load & Merge
# =============================================================================
# Three source files:
#   train.csv    — weekly sales per (Store, Dept, Date)
#   features.csv — macro / weather / promotional variables per (Store, Date)
#   stores.csv   — static metadata (Type, Size) per Store

train    = pd.read_csv("data/train.csv",    parse_dates=["Date"])
features = pd.read_csv("data/features.csv", parse_dates=["Date"])
stores   = pd.read_csv("data/stores.csv")

# Merge on Store + Date (features already has IsHoliday, so drop the duplicate
# from train before merging to keep one authoritative column)
df = (
    train
    .merge(features.drop(columns=["IsHoliday"]), on=["Store", "Date"], how="left")
    .merge(stores, on="Store", how="left")
)

print("Master dataframe shape:", df.shape)
print(df.dtypes)
print(df.head(3).to_string())
print("\nMissing values:\n", df.isnull().sum())

# Drop rows with negative weekly sales — a small number exist and are likely
# returns or accounting adjustments that would distort trend analysis
df = df[df["Weekly_Sales"] >= 0].copy()

# Add convenience time columns used in multiple plots
df["Year"]  = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"]  = df["Date"].dt.isocalendar().week.astype(int)

# =============================================================================
# 2. Overall Weekly Sales Trend
# =============================================================================
# Aggregate across all stores and departments to see the macro trend
weekly_total = df.groupby("Date")["Weekly_Sales"].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(weekly_total["Date"], weekly_total["Weekly_Sales"] / 1e6,
        linewidth=1.4, color="steelblue")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
ax.set_title("Overall Weekly Sales Trend (All Stores & Departments)", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Total Weekly Sales ($ Millions)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_overall_weekly_trend.png", dpi=150)
plt.close()
print("Saved: overall weekly trend")

# =============================================================================
# 3. Sales by Store Type (A, B, C)
# =============================================================================
# Walmart classifies stores into three types by size/format.
# We want to see both the total volume and per-store average.

type_total = df.groupby("Type")["Weekly_Sales"].sum().reset_index()
type_avg   = df.groupby(["Type", "Date"])["Weekly_Sales"].sum().groupby(level=0).mean().reset_index()
type_avg.columns = ["Type", "Avg_Weekly_Sales_Per_Store"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: total sales by type
axes[0].bar(type_total["Type"], type_total["Weekly_Sales"] / 1e9,
            color=sns.color_palette("Set2", 3))
axes[0].set_title("Total Sales by Store Type")
axes[0].set_ylabel("Total Sales ($ Billions)")
axes[0].set_xlabel("Store Type")
for i, row in type_total.iterrows():
    axes[0].text(i, row["Weekly_Sales"] / 1e9 + 0.05,
                 f"${row['Weekly_Sales']/1e9:.1f}B", ha="center", fontsize=10)

# Right: average weekly sales per store (normalised by store count)
store_counts = stores.groupby("Type").size().reset_index(name="Count")
type_total   = type_total.merge(store_counts, on="Type")
type_total["Avg_Per_Store"] = type_total["Weekly_Sales"] / type_total["Count"] / 1e6

axes[1].bar(type_total["Type"], type_total["Avg_Per_Store"],
            color=sns.color_palette("Set2", 3))
axes[1].set_title("Avg Weekly Sales per Store by Type")
axes[1].set_ylabel("Avg Weekly Sales ($ Millions)")
axes[1].set_xlabel("Store Type")

plt.suptitle("Sales by Store Type", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_sales_by_store_type.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: sales by store type")

# =============================================================================
# 4. Effect of Holidays on Sales
# =============================================================================
# IsHoliday flags four Walmart super-events: Super Bowl, Labor Day,
# Thanksgiving, Christmas.  We compare average weekly sales on holiday vs
# non-holiday weeks both overall and by store type.

holiday_avg = df.groupby("IsHoliday")["Weekly_Sales"].mean().reset_index()
holiday_avg["Label"] = holiday_avg["IsHoliday"].map({True: "Holiday", False: "Non-Holiday"})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: simple bar comparison
colors = ["#4CAF50" if h else "#2196F3" for h in holiday_avg["IsHoliday"]]
axes[0].bar(holiday_avg["Label"], holiday_avg["Weekly_Sales"],
            color=colors, width=0.4)
axes[0].set_title("Avg Weekly Sales: Holiday vs Non-Holiday")
axes[0].set_ylabel("Avg Weekly Sales ($)")
for i, row in holiday_avg.iterrows():
    axes[0].text(i, row["Weekly_Sales"] + 200,
                 f"${row['Weekly_Sales']:,.0f}", ha="center", fontsize=10)

# Right: breakdown by store type
type_holiday = df.groupby(["Type", "IsHoliday"])["Weekly_Sales"].mean().reset_index()
type_holiday["Label"] = type_holiday["IsHoliday"].map({True: "Holiday", False: "Non-Holiday"})
sns.barplot(data=type_holiday, x="Type", y="Weekly_Sales", hue="Label",
            ax=axes[1], palette=["#4CAF50", "#2196F3"])
axes[1].set_title("Holiday Effect by Store Type")
axes[1].set_ylabel("Avg Weekly Sales ($)")
axes[1].set_xlabel("Store Type")

plt.suptitle("Holiday Impact on Weekly Sales", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_holiday_effect.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: holiday effect")

# =============================================================================
# 5. Correlation Heatmap — Macro Variables vs Weekly Sales
# =============================================================================
# We check how external factors (temperature, fuel price, CPI, unemployment,
# markdowns) correlate with weekly sales to guide feature selection later.

corr_cols = ["Weekly_Sales", "Temperature", "Fuel_Price",
             "CPI", "Unemployment",
             "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
corr_df = df[corr_cols].dropna()
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))   # show only lower triangle
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5,
            ax=ax, annot_kws={"size": 9})
ax.set_title("Correlation Matrix: Macro Variables vs Weekly Sales", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: correlation heatmap")

# =============================================================================
# 6. Top 10 Departments by Total Sales
# =============================================================================
dept_sales = (
    df.groupby("Dept")["Weekly_Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.barh(dept_sales["Dept"].astype(str), dept_sales["Weekly_Sales"] / 1e9,
               color=sns.color_palette("Blues_r", 10))
ax.set_xlabel("Total Sales ($ Billions)")
ax.set_title("Top 10 Departments by Total Sales", fontsize=13)
ax.invert_yaxis()
for bar, val in zip(bars, dept_sales["Weekly_Sales"] / 1e9):
    ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
            f"${val:.2f}B", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_top10_departments.png", dpi=150)
plt.close()
print("Saved: top 10 departments")

# =============================================================================
# 7. Monthly & Seasonal Patterns
# =============================================================================
# Two views:
#   (a) Average sales by month-of-year — reveals intra-year seasonality
#   (b) Average sales by week-of-year  — higher resolution seasonality

month_avg = df.groupby("Month")["Weekly_Sales"].mean().reset_index()
week_avg  = df.groupby("Week")["Weekly_Sales"].mean().reset_index()

month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, axes = plt.subplots(2, 1, figsize=(13, 9))

# Monthly
axes[0].bar(month_avg["Month"], month_avg["Weekly_Sales"],
            color=sns.color_palette("viridis", 12))
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(month_names)
axes[0].set_title("Avg Weekly Sales by Month (Seasonality)", fontsize=12)
axes[0].set_ylabel("Avg Weekly Sales ($)")

# Weekly
axes[1].plot(week_avg["Week"], week_avg["Weekly_Sales"],
             marker="o", markersize=3, linewidth=1.2, color="darkorange")
axes[1].set_title("Avg Weekly Sales by Week-of-Year", fontsize=12)
axes[1].set_xlabel("Week of Year")
axes[1].set_ylabel("Avg Weekly Sales ($)")

plt.suptitle("Monthly & Seasonal Sales Patterns", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_monthly_seasonal_patterns.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: monthly & seasonal patterns")

# =============================================================================
# 8. Year-over-Year Sales by Store Type
# =============================================================================
yoy = df.groupby(["Year", "Type"])["Weekly_Sales"].sum().reset_index()

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=yoy, x="Year", y="Weekly_Sales", hue="Type",
            palette="Set2", ax=ax)
ax.set_title("Year-over-Year Total Sales by Store Type", fontsize=13)
ax.set_ylabel("Total Sales ($)")
ax.set_xlabel("Year")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_yoy_by_store_type.png", dpi=150)
plt.close()
print("Saved: YoY by store type")

# =============================================================================
# Summary printout
# =============================================================================
print("\n=== EDA Summary ===")
print(f"Date range      : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Stores          : {df['Store'].nunique()}")
print(f"Departments     : {df['Dept'].nunique()}")
print(f"Total rows      : {len(df):,}")
print(f"Holiday weeks % : {df['IsHoliday'].mean()*100:.1f}%")
holiday_uplift = (
    df.groupby("IsHoliday")["Weekly_Sales"].mean()
)
uplift = (holiday_uplift[True] - holiday_uplift[False]) / holiday_uplift[False] * 100
print(f"Holiday sales uplift vs non-holiday: {uplift:+.1f}%")
print(f"\nTop dept by sales: Dept {dept_sales.iloc[0]['Dept']} "
      f"(${dept_sales.iloc[0]['Weekly_Sales']/1e9:.2f}B total)")
print(f"\nAll EDA plots saved to: {PLOT_DIR}/")
