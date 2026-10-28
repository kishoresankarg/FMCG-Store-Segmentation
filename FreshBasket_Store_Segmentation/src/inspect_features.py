
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("data/processed")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

features = pd.read_csv(DATA / "store_features_initial.csv")

# A. Inspect the stores with the highest and lowest growth
print("\n===== HIGHEST GROWTH STORES =====")
print(
    features.nlargest(10, "growth_first_vs_last_13_weeks")[
        ["STORE_CODE", "growth_first_vs_last_13_weeks",
         "total_spend", "active_weeks", "STORE_FORMAT", "STORE_REGION"]
    ].to_string(index=False)
)

print("\n===== LOWEST GROWTH STORES =====")
print(
    features.nsmallest(10, "growth_first_vs_last_13_weeks")[
        ["STORE_CODE", "growth_first_vs_last_13_weeks",
         "total_spend", "active_weeks", "STORE_FORMAT", "STORE_REGION"]
    ].to_string(index=False)
)

# B. Check correlations among numerical features
numeric = features.select_dtypes(include=np.number)
corr = numeric.corr()

print("\n===== STRONG CORRELATIONS (absolute value >= 0.80) =====")
for i, col1 in enumerate(corr.columns):
    for col2 in corr.columns[i + 1:]:
        value = corr.loc[col1, col2]
        if abs(value) >= 0.80:
            print(f"{col1} vs {col2}: {value:.3f}")

corr.to_csv(REPORTS / "feature_correlations.csv")

# C. Check which stores have fewer observed weeks
print("\n===== ACTIVE WEEK COUNTS =====")
print(features["active_weeks"].describe())
print("Stores with fewer than 100 active weeks:",
      (features["active_weeks"] < 100).sum())

# D. Save the outlier review file
features.sort_values(
    "growth_first_vs_last_13_weeks",
    ascending=False
).to_csv(REPORTS / "growth_outlier_review.csv", index=False)

print("\nSaved reports/growth_outlier_review.csv")
print("Saved reports/feature_correlations.csv")
