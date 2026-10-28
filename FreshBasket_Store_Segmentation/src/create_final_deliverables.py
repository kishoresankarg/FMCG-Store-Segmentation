
from pathlib import Path
import pandas as pd

DATA = Path("data/processed")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

assignments = pd.read_csv(DATA / "kmeans_2_assignments.csv")
stores = pd.read_csv(DATA / "store_features_enriched.csv")
categories = pd.read_csv(DATA / "store_category_mix.csv")

# --------------------------------------------------
# 1. Final store assignments
# --------------------------------------------------
final = assignments.merge(stores, on="STORE_CODE", how="left")

# Verify the expected two clusters.
found = sorted(final["cluster"].dropna().unique())
if len(found) != 2:
    raise ValueError(f"Expected 2 clusters, found: {found}")

# Name segments based on their observed median weekly spend.
median_spend = final.groupby("cluster")["avg_weekly_spend"].median()
ordered_clusters = median_spend.sort_values().index.tolist()

segment_names = {
    ordered_clusters[0]: "Smaller-Spend Stores",
    ordered_clusters[1]: "Larger-Spend Stores",
}

final["segment_name"] = final["cluster"].map(segment_names)

final.to_csv(DATA / "final_store_segment_assignments.csv", index=False)

# --------------------------------------------------
# 2. Numeric profiles
# --------------------------------------------------
profile = final.groupby(["cluster", "segment_name"]).agg(
    store_count=("STORE_CODE", "count"),
    median_weekly_spend=("avg_weekly_spend", "median"),
    mean_weekly_spend=("avg_weekly_spend", "mean"),
    median_weekly_volatility=("weekly_spend_volatility", "median"),
    median_relative_volatility=("relative_spend_volatility", "median"),
    median_active_weeks=("active_weeks", "median"),
    stores_with_zero_starting_spend=("zero_starting_spend", "sum"),
).reset_index()

profile["store_percentage"] = (
    profile["store_count"] / len(final) * 100
)

profile.to_csv(REPORTS / "final_segment_profile.csv", index=False)

# --------------------------------------------------
# 3. Store format and region mix
# --------------------------------------------------
pd.crosstab(
    final["segment_name"],
    final["STORE_FORMAT"],
    normalize="index",
).mul(100).round(2).to_csv(
    REPORTS / "final_segment_format_mix.csv"
)

pd.crosstab(
    final["segment_name"],
    final["STORE_REGION"],
    normalize="index",
).mul(100).round(2).to_csv(
    REPORTS / "final_segment_region_mix.csv"
)

# --------------------------------------------------
# 4. Full category mix by segment
# --------------------------------------------------
segment_categories = categories.merge(
    final[["STORE_CODE", "segment_name"]],
    on="STORE_CODE",
    how="inner",
)

category_profile = (
    segment_categories.groupby(
        ["segment_name", "PROD_CODE_10"]
    )
    .agg(
        mean_category_share=("category_share", "mean"),
        total_category_spend=("spend", "sum"),
        stores_observed=("STORE_CODE", "nunique"),
    )
    .reset_index()
)

category_profile.to_csv(
    REPORTS / "final_segment_category_profile.csv",
    index=False,
)

# --------------------------------------------------
# 5. Recommendation matrix
# --------------------------------------------------
recommendations = pd.DataFrame([
    {
        "segment": "Smaller-Spend Stores",
        "observed_evidence": (
            "Lower median weekly spend; predominantly SS format; "
            "higher median relative spend volatility."
        ),
        "recommended_action": (
            "Pilot a focused assortment and replenishment review; "
            "test smaller, targeted promotional campaigns."
        ),
        "measure_to_track": (
            "Weekly spend, units sold, category mix, and stock availability "
            "if inventory data becomes available."
        ),
        "caution": (
            "Current data does not establish stockouts, promotion response, "
            "profitability, or customer demand causes."
        ),
    },
    {
        "segment": "Larger-Spend Stores",
        "observed_evidence": (
            "Higher median weekly spend; predominantly LS format; "
            "lower median relative spend volatility."
        ),
        "recommended_action": (
            "Review high-contribution categories and test category-specific "
            "assortment and replenishment planning."
        ),
        "measure_to_track": (
            "Weekly spend, units sold, category mix, and stock availability "
            "if inventory data becomes available."
        ),
        "caution": (
            "Higher spend does not necessarily mean higher profit; margin, "
            "inventory, and promotion data are not currently available."
        ),
    },
])

recommendations.to_csv(
    REPORTS / "segment_recommendation_matrix.csv",
    index=False,
)

print("Final assignments:", DATA / "final_store_segment_assignments.csv")
print("Segment profile:", REPORTS / "final_segment_profile.csv")
print("Format mix:", REPORTS / "final_segment_format_mix.csv")
print("Region mix:", REPORTS / "final_segment_region_mix.csv")
print("Category profile:", REPORTS / "final_segment_category_profile.csv")
print("Recommendations:", REPORTS / "segment_recommendation_matrix.csv")

print("\n===== FINAL SEGMENT PROFILE =====")
print(profile.round(2).to_string(index=False))
