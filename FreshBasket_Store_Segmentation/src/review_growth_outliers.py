
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/store_week_sales.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# Stores with unusually high or low growth
target_stores = [
    "STORE01856",
    "STORE00962",
    "STORE01769",
    "STORE02752",
    "STORE00779",
    "STORE01062",
    "STORE02121",
]

df = pd.read_csv(DATA_PATH)

# Basic validation
required = {"STORE_CODE", "SHOP_WEEK", "spend", "units"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["SHOP_WEEK"] = pd.to_numeric(df["SHOP_WEEK"], errors="coerce")
df["spend"] = pd.to_numeric(df["spend"], errors="coerce")
df["units"] = pd.to_numeric(df["units"], errors="coerce")

df = df.dropna(subset=["STORE_CODE", "SHOP_WEEK", "spend"])
df = df.sort_values(["SHOP_WEEK", "STORE_CODE"])

# Use the earliest and latest 13 distinct weeks in the dataset
weeks = sorted(df["SHOP_WEEK"].unique())

if len(weeks) < 26:
    raise ValueError("Not enough distinct weeks to compare 13-week periods.")

first_weeks = weeks[:13]
last_weeks = weeks[-13:]

first = (
    df[df["SHOP_WEEK"].isin(first_weeks)]
    .groupby("STORE_CODE")["spend"]
    .agg(first_13_week_spend="sum", first_weeks_observed="count")
)

last = (
    df[df["SHOP_WEEK"].isin(last_weeks)]
    .groupby("STORE_CODE")["spend"]
    .agg(last_13_week_spend="sum", last_weeks_observed="count")
)

review = first.join(last, how="outer").fillna(0)

review["growth_recalculated"] = (
    (review["last_13_week_spend"] - review["first_13_week_spend"])
    / review["first_13_week_spend"].replace(0, float("nan"))
)

review["first_13_week_avg"] = review["first_13_week_spend"] / 13
review["last_13_week_avg"] = review["last_13_week_spend"] / 13

print("\n===== GROWTH OUTLIER REVIEW =====")
print(
    review.reindex(target_stores)
    .round(3)
    .to_string()
)

print("\n===== STORES WITH ZERO OR VERY LOW STARTING SPEND =====")
print(
    review[
        review["first_13_week_spend"] <= review["first_13_week_spend"].quantile(0.05)
    ]
    .sort_values("first_13_week_spend")
    .head(15)
    .round(3)
    .to_string()
)

review.to_csv(REPORT_DIR / "growth_period_review.csv")
print("\nSaved reports/growth_period_review.csv")
