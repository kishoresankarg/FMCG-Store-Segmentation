
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler

DATA = Path("data/processed")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

stores = pd.read_csv(DATA / "store_features_enriched.csv")
category = pd.read_csv(DATA / "store_category_mix.csv")

# Reproduce the top-15 category selection.
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
        "log_spend_change",
        "zero_starting_spend",
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

all_features = [
    "log_avg_weekly_spend",
    "relative_spend_volatility",
    "log_spend_change",
] + category_cols

no_scale_features = [
    "relative_spend_volatility",
    "log_spend_change",
] + category_cols


def prepare_matrix(columns):
    raw = features[columns].replace([np.inf, -np.inf], np.nan)
    imputed = SimpleImputer(strategy="median").fit_transform(raw)
    return StandardScaler().fit_transform(imputed)


def evaluate_matrix(X, label):
    records = []
    reference_labels = {}

    for k in [2, 3]:
        for seed in [21, 42, 77, 123, 2026]:
            model = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=20,
            )
            labels = model.fit_predict(X)
            reference_labels[(k, seed)] = labels

            records.append({
                "feature_set": label,
                "k": k,
                "seed": seed,
                "silhouette": silhouette_score(
                    X, labels,
                    sample_size=min(3000, len(X)),
                    random_state=42,
                ),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "smallest_cluster": int(
                    pd.Series(labels).value_counts().min()
                ),
                "largest_cluster": int(
                    pd.Series(labels).value_counts().max()
                ),
            })

    # Compare each seed's labels with the seed-42 solution.
    for k in [2, 3]:
        reference = reference_labels[(k, 42)]
        scores = [
            adjusted_rand_score(reference, reference_labels[(k, seed)])
            for seed in [21, 77, 123, 2026]
        ]
        print(
            f"{label} | k={k} | mean ARI vs seed 42: "
            f"{np.mean(scores):.4f} | individual ARIs: "
            f"{[round(s, 4) for s in scores]}"
        )

    return records


records = []
records.extend(evaluate_matrix(prepare_matrix(all_features), "with_store_scale"))
records.extend(
    evaluate_matrix(prepare_matrix(no_scale_features), "without_store_scale")
)

results = pd.DataFrame(records)
results.to_csv(REPORTS / "cluster_stability_sensitivity.csv", index=False)

summary = results.groupby(["feature_set", "k"]).agg(
    silhouette_mean=("silhouette", "mean"),
    silhouette_std=("silhouette", "std"),
    davies_bouldin_mean=("davies_bouldin", "mean"),
    smallest_cluster_mean=("smallest_cluster", "mean"),
    largest_cluster_mean=("largest_cluster", "mean"),
).reset_index()

summary.to_csv(REPORTS / "cluster_sensitivity_summary.csv", index=False)

print("\n===== SENSITIVITY SUMMARY =====")
print(summary.round(4).to_string(index=False))
print("\nSaved reports/cluster_stability_sensitivity.csv")
print("Saved reports/cluster_sensitivity_summary.csv")
