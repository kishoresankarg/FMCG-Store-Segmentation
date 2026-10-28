
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(DATA_DIR / "store_features_initial.csv")
weekly = pd.read_csv(DATA_DIR / "store_week_sales.csv")
category = pd.read_csv(DATA_DIR / "store_category_mix.csv")

print("\n========== DATASET OVERVIEW ==========")
print("Store feature rows:", len(features))
print("Unique stores:", features["STORE_CODE"].nunique())
print("Weekly sales rows:", len(weekly))
print("Category mix rows:", len(category))
print("\nFeature columns:")
print(features.columns.tolist())

# 1. Duplicate checks
print("\n========== DUPLICATES ==========")
print("Duplicate store rows:", features["STORE_CODE"].duplicated().sum())
print(
    "Duplicate store-week rows:",
    weekly.duplicated(["STORE_CODE", "SHOP_WEEK"]).sum()
)

# 2. Missing values
missing = features.isna().sum().to_frame("missing_count")
missing["missing_percent"] = (
    missing["missing_count"] / len(features) * 100
).round(2)

print("\n========== MISSING VALUES ==========")
print(missing)
missing.to_csv(REPORT_DIR / "missing_values.csv")

# 3. Numerical summary
numeric = features.select_dtypes(include=np.number)
summary = numeric.describe().T
summary.to_csv(REPORT_DIR / "feature_summary.csv")
print("\n========== NUMERICAL SUMMARY ==========")
print(summary.round(2))

# 4. Impossible or suspicious values
print("\n========== VALUE CHECKS ==========")
print("Negative total units:", (features["total_units"] < 0).sum())
print("Negative total spend:", (features["total_spend"] < 0).sum())
print("Zero total spend:", (features["total_spend"] == 0).sum())
print("Stores with fewer than 52 active weeks:",
      (features["active_weeks"] < 52).sum())

# 5. Store format and region distribution
for col in ["STORE_FORMAT", "STORE_REGION"]:
    print(f"\n{col} distribution:")
    print(features[col].value_counts(dropna=False))
    features[col].value_counts(dropna=False).to_csv(
        REPORT_DIR / f"{col.lower()}_distribution.csv"
    )

# 6. Sales distributions (log scale helps show skew)
for col in ["total_spend", "total_units", "avg_weekly_spend"]:
    plt.figure(figsize=(8, 5))
    sns.histplot(features[col].dropna(), bins=40)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{col}_distribution.png", dpi=150)
    plt.close()

# 7. Correlation analysis
corr = numeric.corr()
corr.to_csv(REPORT_DIR / "feature_correlations.csv")

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Store Feature Correlations")
plt.tight_layout()
plt.savefig(REPORT_DIR / "feature_correlation_heatmap.png", dpi=150)
plt.close()

# 8. Category-mix checks
print("\n========== CATEGORY MIX ==========")
print("Unique product groups:", category["PROD_CODE_10"].nunique())
print("Missing category shares:", category["category_share"].isna().sum())

share_sums = category.groupby("STORE_CODE")["category_share"].sum()
print("Stores represented in category mix:", len(share_sums))
print("Category shares near 1:",
      np.isclose(share_sums, 1, atol=0.01).sum(), "/", len(share_sums))

print("\nEDA complete. Reports saved to:", REPORT_DIR.resolve())
