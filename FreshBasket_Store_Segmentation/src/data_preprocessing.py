
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = sorted(RAW_DIR.glob("transactions_*.csv"))

if not FILES:
    raise FileNotFoundError(f"No transaction files found in {RAW_DIR}")

weekly_parts = []

for i, file in enumerate(FILES, start=1):
    print(f"[{i}/{len(FILES)}] Processing {file.name}")

    for chunk in pd.read_csv(file, chunksize=100_000):
        # Standardize columns that vary between files
        if "SPEND" not in chunk.columns and "SFNND" in chunk.columns:
            chunk = chunk.rename(columns={"SFNND": "SPEND"})

        if "BASKET_TYPE" not in chunk.columns and "BASKET_TYFN" in chunk.columns:
            chunk = chunk.rename(columns={"BASKET_TYFN": "BASKET_TYPE"})

        required = [
            "SHOP_WEEK", "STORE_CODE", "PROD_CODE",
            "QUANTITY", "SPEND"
        ]
        missing = set(required) - set(chunk.columns)

        if missing:
            raise ValueError(
                f"{file.name} is missing required columns: {sorted(missing)}"
            )

        chunk["QUANTITY"] = pd.to_numeric(
            chunk["QUANTITY"], errors="coerce"
        )
        chunk["SPEND"] = pd.to_numeric(
            chunk["SPEND"], errors="coerce"
        )

        # Remove rows that cannot contribute to store/product sales
        chunk = chunk.dropna(
            subset=["SHOP_WEEK", "STORE_CODE", "PROD_CODE",
                    "QUANTITY", "SPEND"]
        )

        # Aggregate within each chunk to reduce memory usage
        group_cols = [
            "SHOP_WEEK", "STORE_CODE", "PROD_CODE"
        ]

        part = chunk.groupby(
            group_cols, as_index=False, observed=True
        ).agg(
            units=("QUANTITY", "sum"),
            revenue=("SPEND", "sum"),
            transaction_lines=("PROD_CODE", "size")
        )

        weekly_parts.append(part)

# Combine partial aggregates and aggregate again because the
# same store/week/product may appear in multiple chunks.
weekly = pd.concat(weekly_parts, ignore_index=True)

weekly = weekly.groupby(
    ["SHOP_WEEK", "STORE_CODE", "PROD_CODE"],
    as_index=False
).agg(
    units=("units", "sum"),
    revenue=("revenue", "sum"),
    transaction_lines=("transaction_lines", "sum")
)

output_file = OUTPUT_DIR / "store_week_product_sales.csv"
weekly.to_csv(output_file, index=False)

print("\nProcessing complete.")
print("Aggregated rows:", len(weekly))
print("Unique stores:", weekly["STORE_CODE"].nunique())
print("Unique products:", weekly["PROD_CODE"].nunique())
print("Output:", output_file)
