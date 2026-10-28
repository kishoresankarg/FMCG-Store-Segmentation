
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA = Path("data/processed")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

stores = pd.read_csv(DATA / "store_features_enriched.csv")
category = pd.read_csv(DATA / "store_category_mix.csv")

# Reproduce the same top-15 category features used in clustering.
top_categories = (
    category.groupby("PROD_CODE_10")["spend"]
    .sum()
    .nlargest(15)
    .index
)

cat_wide = category[
    category["PROD_CODE_10"].isin(top_categories)
].pivot_table(
    index="STORE_CODE",
    columns="PROD_CODE_10",
    values="category_share",
    aggfunc="sum",
    fill_value=0,
)

cat_wide.columns = [f"category_share_{c}" for c in cat_wide.columns]
cat_wide = cat_wide.reset_index()

features = stores[
    [
        "STORE_CODE",
        "avg_weekly_spend",
        "weekly_spend_volatility",
        "active_weeks",
        "log_spend_change",
        "zero_starting_spend",
        "STORE_FORMAT",
        "STORE_REGION",
    ]
].copy()

features["log_avg_weekly_spend"] = np.log1p(
    features["avg_weekly_spend"].clip(lower=0)
)
features["relative_spend_volatility"] = (
    features["weekly_spend_volatility"]
    / features["avg_weekly_spend"].replace(0, np.nan)
)

features.loc[
    features["zero_starting_spend"].eq(1), "log_spend_change"
] = np.nan

features = features.merge(cat_wide, on="STORE_CODE", how="left")

category_cols = [
    c for c in features.columns if c.startswith("category_share_")
]
numeric_cols = [
    "log_avg_weekly_spend",
    "relative_spend_volatility",
    "log_spend_change",
] + category_cols

X = features[numeric_cols].replace([np.inf, -np.inf], np.nan)
X = SimpleImputer(strategy="median").fit_transform(X)
X = StandardScaler().fit_transform(X)

# Fit both candidates using the same preprocessing.
for k in [2, 3]:
    labels = KMeans(
        n_clusters=k, random_state=42, n_init=20
    ).fit_predict(X)

    result = features.copy()
    result["cluster"] = labels

    # Store-level numeric profile
    numeric_profile = result.groupby("cluster").agg(
        store_count=("STORE_CODE", "count"),
        avg_weekly_spend=("avg_weekly_spend", "median"),
        median_weekly_spend=("avg_weekly_spend", "median"),
        median_relative_volatility=("relative_spend_volatility", "median"),
        median_active_weeks=("active_weeks", "median"),
        median_log_spend_change=("log_spend_change", "median"),
        zero_starting_spend_pct=("zero_starting_spend", "mean"),
        short_history_pct=("active_weeks", lambda x: (x < 100).mean()),
    )

    numeric_profile["zero_starting_spend_pct"] *= 100
    numeric_profile["short_history_pct"] *= 100
    numeric_profile.to_csv(
        REPORTS / f"kmeans_{k}_cluster_numeric_profile.csv"
    )

    # Store format and region mix
    pd.crosstab(
        result["cluster"], result["STORE_FORMAT"], normalize="index"
    ).to_csv(REPORTS / f"kmeans_{k}_format_mix.csv")

    pd.crosstab(
        result["cluster"], result["STORE_REGION"], normalize="index"
    ).to_csv(REPORTS / f"kmeans_{k}_region_mix.csv")

    # Average category shares by cluster
    result.groupby("cluster")[category_cols].mean().to_csv(
        REPORTS / f"kmeans_{k}_category_profile.csv"
    )

    # Save assignments for this candidate
    result[["STORE_CODE", "cluster"]].to_csv(
        DATA / f"kmeans_{k}_assignments.csv", index=False
    )

    print(f"\n===== K-MEANS {k} CLUSTERS =====")
    print("Cluster sizes:")
    print(result["cluster"].value_counts().sort_index().to_string())
    print("\nNumeric profile:")
    print(numeric_profile.round(3).to_string())
    print("\nStore format mix (%):")
    print(
        (pd.crosstab(
            result["cluster"], result["STORE_FORMAT"], normalize="index"
        ) * 100).round(1).to_string()
    )

print("\nSaved profiles and assignments under data/processed and reports/")
