
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/processed")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

weekly = pd.read_csv(DATA / "store_week_sales.csv")
base = pd.read_csv(DATA / "store_features_initial.csv")

weekly["SHOP_WEEK"] = pd.to_numeric(weekly["SHOP_WEEK"], errors="coerce")
weekly["spend"] = pd.to_numeric(weekly["spend"], errors="coerce")

weekly = weekly.dropna(subset=["STORE_CODE", "SHOP_WEEK", "spend"])
weekly = weekly.sort_values(["STORE_CODE", "SHOP_WEEK"])

weeks = sorted(weekly["SHOP_WEEK"].unique())
first_weeks = weeks[:13]
last_weeks = weeks[-13:]

first = (
    weekly[weekly["SHOP_WEEK"].isin(first_weeks)]
    .groupby("STORE_CODE")["spend"]
    .sum()
    .rename("first_13_week_spend")
)

last = (
    weekly[weekly["SHOP_WEEK"].isin(last_weeks)]
    .groupby("STORE_CODE")["spend"]
    .sum()
    .rename("last_13_week_spend")
)

growth = pd.concat([first, last], axis=1).fillna(0)

# Log change is defined even when one period has zero spend.
# +1 is a small offset to handle zeros.
growth["log_spend_change"] = np.log1p(
    growth["last_13_week_spend"]
) - np.log1p(growth["first_13_week_spend"])

# Track whether the starting period has no recorded spend.
growth["zero_starting_spend"] = (
    growth["first_13_week_spend"] == 0
).astype(int)

# Calculate the number of observed weeks per store.
history = (
    weekly.groupby("STORE_CODE")["SHOP_WEEK"]
    .nunique()
    .rename("observed_weeks")
)

# Merge engineered features into the existing store table.
final = (
    base.merge(growth.reset_index(), on="STORE_CODE", how="left")
        .merge(history.reset_index(), on="STORE_CODE", how="left")
)

final["zero_starting_spend"] = (
    final["zero_starting_spend"].fillna(1).astype(int)
)
final["observed_weeks"] = final["observed_weeks"].fillna(0).astype(int)

# Relative volatility: variability compared with store scale.
final["relative_spend_volatility"] = (
    final["weekly_spend_volatility"]
    / final["avg_weekly_spend"].replace(0, np.nan)
)

# Save a complete feature table for subsequent review.
output = DATA / "store_features_enriched.csv"
final.to_csv(output, index=False)

# Save a focused audit report for the stores of concern.
audit = final.sort_values(
    "log_spend_change", ascending=False
)[
    [
        "STORE_CODE",
        "first_13_week_spend",
        "last_13_week_spend",
        "growth_first_vs_last_13_weeks",
        "log_spend_change",
        "zero_starting_spend",
        "active_weeks",
        "observed_weeks",
        "STORE_FORMAT",
        "STORE_REGION",
    ]
]

audit.to_csv(REPORTS / "growth_feature_audit.csv", index=False)

print("Saved:", output)
print("Saved: reports/growth_feature_audit.csv")
print("\nHighest log-spend changes:")
print(audit.head(10).round(3).to_string(index=False))

print("\nStores with zero starting spend:")
print(int(final["zero_starting_spend"].sum()))

print("\nStores with fewer than 100 active weeks:")
print(int((final["active_weeks"] < 100).sum()))

print("\nFinal feature table shape:", final.shape)
