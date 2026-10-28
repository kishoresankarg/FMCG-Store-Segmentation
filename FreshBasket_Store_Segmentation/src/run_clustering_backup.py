
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA = Path("data/processed")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
MIN_CLUSTERS = 2
MAX_CLUSTERS = 8
TOP_CATEGORY_COUNT = 15

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
stores = pd.read_csv(DATA / "store_features_enriched.csv")
category = pd.read_csv(DATA / "store_category_mix.csv")

required = {
    "STORE_CODE",
    "avg_weekly_spend",
    "weekly_spend_volatility",
    "active_weeks",
    "log_spend_change",
    "zero_starting_spend",
}
missing = required - set(stores.columns)
if missing:
    raise ValueError(f"Missing store feature columns: {sorted(missing)}")

# --------------------------------------------------
# 2. Keep top categories by total spend
#    The complete category file is preserved.
# --------------------------------------------------
top_categories = (
    category.groupby("PROD_CODE_10")["spend"]
    .sum()
    .nlargest(TOP_CATEGORY_COUNT)
    .index
)

cat_top = category[category["PROD_CODE_10"].isin(top_categories)]

cat_wide = cat_top.pivot_table(
    index="STORE_CODE",
    columns="PROD_CODE_10",
    values="category_share",
    aggfunc="sum",
    fill_value=0,
)

cat_wide.columns = [f"category_share_{c}" for c in cat_wide.columns]
cat_wide = cat_wide.reset_index()

# --------------------------------------------------
# 3. Engineer robust store-level features
# --------------------------------------------------
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

# Do not use log growth as if it were comparable for stores
# with zero starting spend. Their growth is unknown.
features.loc[
    features["zero_starting_spend"].eq(1), "log_spend_change"
] = np.nan

# Keep history as a quality flag, not as a main segmentation dimension.
features["short_history_flag"] = (
    features["active_weeks"] < 100
).astype(int)

features = features.merge(cat_wide, on="STORE_CODE", how="left")

category_cols = [
    c for c in features.columns if c.startswith("category_share_")
]

numeric_cols = [
    "log_avg_weekly_spend",
    "relative_spend_volatility",
    "log_spend_change",
] + category_cols

# --------------------------------------------------
# 4. Impute and scale
# --------------------------------------------------
X_raw = features[numeric_cols].replace([np.inf, -np.inf], np.nan)

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("Stores:", len(features))
print("Clustering features:", len(numeric_cols))
print("Top category features:", len(category_cols))
print("Stores with unknown starting-period growth:",
      int(features["zero_starting_spend"].sum()))

# --------------------------------------------------
# 5. Compare clustering algorithms and cluster counts
# --------------------------------------------------
records = []
models = {}

for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
    candidates = {
        "KMeans": KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        ),
        "Agglomerative": AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
        ),
    }

    for model_name, model in candidates.items():
        labels = model.fit_predict(X_scaled)
        n_labels = len(set(labels))

        if n_labels < 2 or n_labels >= len(labels):
            continue

        records.append({
            "algorithm": model_name,
            "k": k,
            "silhouette": silhouette_score(
                X_scaled, labels, sample_size=min(3000, len(X_scaled)),
                random_state=RANDOM_STATE
            ),
            "davies_bouldin": davies_bouldin_score(X_scaled, labels),
            "calinski_harabasz": calinski_harabasz_score(
                X_scaled, labels
            ),
            "smallest_cluster": int(pd.Series(labels).value_counts().min()),
            "largest_cluster": int(pd.Series(labels).value_counts().max()),
        })

        models[(model_name, k)] = (model, labels)

comparison = pd.DataFrame(records).sort_values(
    ["silhouette", "davies_bouldin"],
    ascending=[False, True],
)

comparison.to_csv(REPORTS / "clustering_model_comparison.csv", index=False)

print("\n===== MODEL COMPARISON =====")
print(comparison.round(4).to_string(index=False))

# Select the best silhouette score as an initial candidate.
# Business interpretability and tiny clusters must still be reviewed.
best = comparison.iloc[0]
best_key = (best["algorithm"], int(best["k"]))
best_model, best_labels = models[best_key]

features["cluster"] = best_labels
features.to_csv(DATA / "store_cluster_assignments.csv", index=False)

print("\nInitial best model:", best_key)
print("Cluster sizes:")
print(features["cluster"].value_counts().sort_index().to_string())

# --------------------------------------------------
# 6. Cluster profiles and category profiles
# --------------------------------------------------
profile_cols = [
    "avg_weekly_spend",
    "weekly_spend_volatility",
    "relative_spend_volatility",
    "active_weeks",
    "log_spend_change",
    "zero_starting_spend",
    "short_history_flag",
]

profiles = features.groupby("cluster")[profile_cols].agg(
    ["mean", "median", "count"]
)
profiles.to_csv(REPORTS / "cluster_profiles.csv")

format_profile = pd.crosstab(
    features["cluster"], features["STORE_FORMAT"], normalize="index"
)
format_profile.to_csv(REPORTS / "cluster_store_format_mix.csv")

region_profile = pd.crosstab(
    features["cluster"], features["STORE_REGION"], normalize="index"
)
region_profile.to_csv(REPORTS / "cluster_store_region_mix.csv")

if category_cols:
    category_profile = features.groupby("cluster")[category_cols].mean()
    category_profile.to_csv(REPORTS / "cluster_category_mix.csv")

print("\nSaved data/processed/store_cluster_assignments.csv")
print("Saved reports/clustering_model_comparison.csv")
print("Saved reports/cluster_profiles.csv")
print("Saved reports/cluster_store_format_mix.csv")
print("Saved reports/cluster_store_region_mix.csv")
print("Saved reports/cluster_category_mix.csv")
