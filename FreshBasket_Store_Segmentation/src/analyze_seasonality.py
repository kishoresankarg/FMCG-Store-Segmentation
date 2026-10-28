from pathlib import Path
import pandas as pd

data_path = Path("data/processed/store_week_sales.csv")
report_dir = Path("reports")
report_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)

required = {"STORE_CODE", "SHOP_WEEK", "spend"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df["SHOP_WEEK"] = pd.to_numeric(df["SHOP_WEEK"], errors="coerce")
df["spend"] = pd.to_numeric(df["spend"], errors="coerce")
df = df.dropna(subset=["STORE_CODE", "SHOP_WEEK", "spend"]).copy()
df["SHOP_WEEK"] = df["SHOP_WEEK"].astype(int)

# Interpret SHOP_WEEK as ISO year + ISO week, then validate.
week_text = df["SHOP_WEEK"].astype(str).str.zfill(6)
df["week_start"] = pd.to_datetime(
    week_text + "1", format="%G%V%u", errors="coerce"
)

if df["week_start"].isna().any():
    bad = df.loc[df["week_start"].isna(), "SHOP_WEEK"].unique()
    raise ValueError(f"Invalid ISO-style week codes: {bad[:10]}")

weekly = (
    df.groupby(["SHOP_WEEK", "week_start"], as_index=False)
    .agg(
        total_spend=("spend", "sum"),
        active_stores=("STORE_CODE", "nunique"),
        store_week_rows=("STORE_CODE", "size"),
    )
    .sort_values("week_start")
)

expected = pd.date_range(
    weekly["week_start"].min(),
    weekly["week_start"].max(),
    freq="W-MON",
)
observed = pd.DatetimeIndex(weekly["week_start"].drop_duplicates())
missing_weeks = expected.difference(observed)

weekly["iso_year"] = weekly["week_start"].dt.isocalendar().year.astype(int)
weekly["iso_week"] = weekly["week_start"].dt.isocalendar().week.astype(int)
weekly["spend_per_active_store"] = (
    weekly["total_spend"] / weekly["active_stores"].replace(0, float("nan"))
)

weekly.to_csv(report_dir / "seasonality_weekly_summary.csv", index=False)

monthly = (
    weekly.assign(month=weekly["week_start"].dt.to_period("M").astype(str))
    .groupby("month", as_index=False)
    .agg(
        total_weekly_spend=("total_spend", "sum"),
        average_weekly_spend=("total_spend", "mean"),
        average_active_stores=("active_stores", "mean"),
        average_spend_per_active_store=("spend_per_active_store", "mean"),
        weeks_in_month=("SHOP_WEEK", "count"),
    )
)
monthly.to_csv(report_dir / "seasonality_monthly_summary.csv", index=False)

week_profile = (
    weekly.groupby("iso_week", as_index=False)
    .agg(
        average_weekly_spend=("total_spend", "mean"),
        average_active_stores=("active_stores", "mean"),
        average_spend_per_active_store=("spend_per_active_store", "mean"),
        years_observed=("iso_year", "nunique"),
        observations=("SHOP_WEEK", "count"),
    )
)
week_profile.to_csv(report_dir / "seasonality_week_of_year_summary.csv", index=False)

year_summary = (
    weekly.groupby("iso_year", as_index=False)
    .agg(
        weeks_observed=("SHOP_WEEK", "count"),
        total_spend=("total_spend", "sum"),
        average_weekly_spend=("total_spend", "mean"),
        average_active_stores=("active_stores", "mean"),
        average_spend_per_active_store=("spend_per_active_store", "mean"),
    )
)
year_summary.to_csv(report_dir / "seasonality_year_summary.csv", index=False)

print("=== SEASONALITY DATA AUDIT ===")
print("Source:", data_path)
print("Weekly observations:", len(weekly))
print("Unique calendar weeks:", weekly["week_start"].nunique())
print("First week:", weekly["week_start"].min().date())
print("Last week:", weekly["week_start"].max().date())
print("Expected weeks in date range:", len(expected))
print("Missing calendar weeks:", len(missing_weeks))
if len(missing_weeks):
    print("Missing week starts:", [str(x.date()) for x in missing_weeks])

print("\n=== WEEKS PER YEAR ===")
print(year_summary.to_string(index=False))

print("\n=== MONTHLY SUMMARY (first 12 rows) ===")
print(monthly.head(12).to_string(index=False))

print("\n=== HIGHEST AND LOWEST SPEND WEEKS ===")
print(
    weekly.nlargest(5, "total_spend")[
        ["week_start", "total_spend", "active_stores",
         "spend_per_active_store"]
    ].to_string(index=False)
)
print(
    weekly.nsmallest(5, "total_spend")[
        ["week_start", "total_spend", "active_stores",
         "spend_per_active_store"]
    ].to_string(index=False)
)

print("\nReports saved:")
for name in [
    "seasonality_weekly_summary.csv",
    "seasonality_monthly_summary.csv",
    "seasonality_week_of_year_summary.csv",
    "seasonality_year_summary.csv",
]:
    print(" -", report_dir / name)
