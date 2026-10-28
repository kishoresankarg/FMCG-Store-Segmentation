
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(RAW_DIR.glob("transactions_*.csv"))
if not files:
    raise FileNotFoundError("No transaction CSV files found.")

# Compact intermediate aggregations
weekly_parts = []
category_parts = []
store_parts = []

for i, file in enumerate(files, start=1):
    print(f"[{i}/{len(files)}] {file.name}")

    for df in pd.read_csv(file, chunksize=100_000):
        if "SPEND" not in df.columns and "SFNND" in df.columns:
            df = df.rename(columns={"SFNND": "SPEND"})

        df["SHOP_DATE"] = pd.to_datetime(
            df["SHOP_DATE"].astype("Int64").astype(str),
            format="%Y%m%d",
            errors="coerce"
        )

        for col in ["QUANTITY", "SPEND"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(
            subset=["STORE_CODE", "SHOP_DATE", "QUANTITY", "SPEND"]
        )

        # Store-week sales: used for growth and volatility
        weekly = df.groupby(
            ["STORE_CODE", "SHOP_WEEK"], as_index=False
        ).agg(
            units=("QUANTITY", "sum"),
            spend=("SPEND", "sum"),
            active_days=("SHOP_DATE", "nunique")
        )
        weekly_parts.append(weekly)

        # Product hierarchy mix: PROD_CODE_10 is a product group
        if "PROD_CODE_10" in df.columns:
            cat = df.dropna(subset=["PROD_CODE_10"]).groupby(
                ["STORE_CODE", "PROD_CODE_10"], as_index=False
            ).agg(spend=("SPEND", "sum"), units=("QUANTITY", "sum"))
            category_parts.append(cat)

        # Store attributes observed in the transaction data
        cols = [
            c for c in ["STORE_CODE", "STORE_FORMAT", "STORE_REGION"]
            if c in df.columns
        ]
        store_parts.append(df[cols].drop_duplicates())

# Combine partial groups from chunks and files
weekly = pd.concat(weekly_parts, ignore_index=True)
weekly = weekly.groupby(
    ["STORE_CODE", "SHOP_WEEK"], as_index=False
).agg(
    units=("units", "sum"),
    spend=("spend", "sum"),
    active_days=("active_days", "max")
)

weekly.to_csv(OUT_DIR / "store_week_sales.csv", index=False)

# Overall store-level sales features
features = weekly.groupby("STORE_CODE").agg(
    total_units=("units", "sum"),
    total_spend=("spend", "sum"),
    avg_weekly_spend=("spend", "mean"),
    median_weekly_spend=("spend", "median"),
    weekly_spend_volatility=("spend", "std"),
    active_weeks=("SHOP_WEEK", "nunique")
).reset_index()

features["weekly_spend_volatility"] = (
    features["weekly_spend_volatility"].fillna(0)
)

# Growth: compare first and last 13 observed weeks per store
growth_rows = []
for store, group in weekly.groupby("STORE_CODE"):
    group = group.sort_values("SHOP_WEEK")
    first = group.head(13)["spend"].mean()
    last = group.tail(13)["spend"].mean()

    growth = np.nan
    if pd.notna(first) and first > 0:
        growth = (last - first) / first

    growth_rows.append({
        "STORE_CODE": store,
        "growth_first_vs_last_13_weeks": growth
    })

features = features.merge(
    pd.DataFrame(growth_rows), on="STORE_CODE", how="left"
)

# Category mix: share of each store's spend by product group
if category_parts:
    category = pd.concat(category_parts, ignore_index=True)
    category = category.groupby(
        ["STORE_CODE", "PROD_CODE_10"], as_index=False
    ).agg(spend=("spend", "sum"))

    category["category_share"] = category["spend"] / category.groupby(
        "STORE_CODE"
    )["spend"].transform("sum")

    category.to_csv(OUT_DIR / "store_category_mix.csv", index=False)

# One row per store for available descriptive attributes
store_info = pd.concat(store_parts, ignore_index=True).drop_duplicates()
store_info = store_info.groupby("STORE_CODE", as_index=False).agg(
    STORE_FORMAT=("STORE_FORMAT", "first"),
    STORE_REGION=("STORE_REGION", "first")
)

features = features.merge(store_info, on="STORE_CODE", how="left")
features.to_csv(OUT_DIR / "store_features_initial.csv", index=False)

print("\nStore-level feature dataset created.")
print("Stores:", features["STORE_CODE"].nunique())
print("Feature columns:", features.columns.tolist())
print("Saved:", OUT_DIR / "store_features_initial.csv")
print("Saved:", OUT_DIR / "store_week_sales.csv")
print("Saved:", OUT_DIR / "store_category_mix.csv")
