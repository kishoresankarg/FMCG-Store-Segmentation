 FMCG Retail Store Segmentation Using Clustering

1. Project Overview

This project segments Fast-Moving Consumer Goods (FMCG) retail stores using transaction-derived sales features and unsupervised machine learning.

The objective is to identify groups of stores with similar sales scale, sales variability, growth patterns, and product-category mix, then translate the findings into practical business recommendations.

The project includes transaction preprocessing, exploratory data analysis, feature engineering, clustering model comparison, sensitivity analysis, segment profiling, and recommendation generation.

The final two-cluster K-Means solution identifies 139 smaller-spend stores and 622 larger-spend stores. These groups provide a starting point for investigating differences in sales behavior and testing targeted business strategies.

**Important:** The resulting segments describe observed sales behavior. They do not directly measure profitability or establish causal relationships.

2. Dataset

The project uses transaction data provided as multiple CSV files.

| Dataset characteristic | Reported value |
|---|---:|
| Raw transaction files | 117 |
| Approximate transaction records | 31 million |
| Unique stores analyzed | 761 |
| Distinct products in processed data | 4,997 |
| Product-category groups | 250 |
| Store-week records | 88,551 |
| Store-category records | 173,765 |
| Weekly observations | 117 |

The weekly aggregated dataset was inspected as covering **13 February 2006 to 5 May 2008**, with 117 consecutive calendar weeks. The raw transaction date range should be verified independently before publishing a definitive date range for the original input files.

Relevant fields include store identifiers, transaction dates, quantities, spend-related fields, store format, store region, and product-category codes.

The `SFNND` field was normalized to `SPEND` during preprocessing. Its exact business definition and currency were not independently verified.

Data limitations

The available data does not establish reliable product margins, promotion effectiveness, inventory stockouts, or store-level selling-price differences. These factors are not treated as measured clustering features.

3. Project Objectives

The main objectives are:

1. Inspect input schemas and validate data quality.
2. Preprocess large transaction files efficiently.
3. Aggregate transactions into store-week and store-category datasets.
4. Engineer meaningful store-level features.
5. Analyze sales distributions, correlations, seasonality, and unusual growth values.
6. Compare K-Means and Agglomerative Clustering across different cluster counts.
7. Evaluate clustering quality, cluster sizes, and sensitivity.
8. Profile the resulting store segments.
9. Prepare actionable business recommendations.
10. Generate reproducible assignments, reports, and final deliverables.

4. Methodology

The project follows this workflow:

1. Inspect the input files and their schemas.
2. Standardize inconsistent column names.
3. Process transaction records in chunks to reduce memory usage.
4. Aggregate transactions into store-week-product, store-week, and store-category datasets.
5. Build initial store-level features.
6. Analyze feature distributions, correlations, data quality, and growth outliers.
7. Generate enriched features.
8. Impute missing numerical values and standardize clustering features.
9. Compare K-Means and Agglomerative Clustering for cluster counts from two through eight.
10. Evaluate the models using clustering metrics and cluster-size information.
11. Test sensitivity to random initialization and selected feature changes.
12. Profile the selected segments and generate business recommendations.

The workflow separates data preparation, clustering, interpretation, and final deliverable generation into individual scripts.

5. Exploratory Data Analysis

The initial analysis examined store-level spend, units, weekly variability, activity, store format, region, and product-category mix.

Reported summary statistics include:

| Feature | Reported value |
|---|---:|
| Mean total units per store | Approximately 61,509 |
| Mean total spend per store | Approximately 78,428 |
| Mean average weekly spend across stores | Approximately 674.08 |
| Mean weekly spend volatility across stores | Approximately 153.69 |
| Mean active weeks | Approximately 116.36 out of 117 |

The reported store-format distribution is:

| Store format | Number of stores |
|---|---:|
| LS | 330 |
| MS | 236 |
| SS | 167 |
| XLS | 28 |
| **Total** | **761** |

The stores span 12 regions.

Correlation analysis

Total spend and average weekly spend had a reported correlation of approximately 0.998. Total units and total spend had a reported correlation of approximately 0.995.

These high correlations indicate that some raw sales features contain overlapping information. The clustering pipeline therefore uses selected and transformed features rather than including every raw measure indiscriminately.

Growth outlier review

Some stores showed unusually large changes between the first and last 13-week comparison periods. These cases were reviewed because low starting-period spend can produce very large relative increases.

Ten stores were identified with zero starting-period spend and positive final-period spend. Conventional percentage growth is undefined for these stores, so their growth values require special treatment.

Extreme changes are flagged for investigation rather than automatically classified as errors or evidence of successful business performance.

6. Exploratory Seasonality Analysis

The aggregated store-week dataset contains 117 consecutive calendar weeks from 13 February 2006 to 5 May 2008, with no missing calendar weeks in the observed range. This provides approximately 27 months of history for exploratory weekly and monthly spending analysis.

Reported average weekly spending was approximately:

| Period | Average weekly spend |
|---|---:|
| 2006 | 504,159 |
| 2007 | 514,312 |
| First 19 weeks of 2008 | 513,055 |

October showed a similar pattern in 2006 and 2007. Average weekly spending was approximately 563,141 in October 2006 and 554,829 in October 2007.

Both years also showed a weekly spending peak around ISO week 43, followed by a sharp decline in week 44.

These observations suggest a pattern worth investigating, but they do not establish a seasonal cause. The available dataset does not establish whether promotions, holidays, pricing changes, reporting effects, or other operational factors explain the observed peaks.

The relatively short observation window and incomplete 2008 calendar year also limit conclusions about annual seasonality.

The analysis is therefore exploratory rather than proof of a recurring seasonal effect.

The generated seasonality summaries are intended to be stored in the `reports/` directory:

- `seasonality_weekly_summary.csv`
- `seasonality_monthly_summary.csv`
- `seasonality_week_of_year_summary.csv`
- `seasonality_year_summary.csv`

7. Feature Engineering

Store-level features were constructed to represent sales scale, variability, changes in spend, and product-category preferences.

7.1 Average weekly spend

Average weekly spend represents the typical recorded sales scale of a store over its observed history.

7.2 Log-transformed average spend

The `log_avg_weekly_spend` feature is calculated as:

`log1p(max(avg_weekly_spend, 0))`

This transformation reduces the influence of highly skewed spend values.

7.3 Relative spend volatility

Relative spend volatility is calculated as:

`weekly_spend_volatility / avg_weekly_spend`

This feature measures weekly spend variability relative to typical spend. When average spend is zero, the ratio is undefined and is handled as a missing value during preprocessing.

7.4 Log spend change

Log spend change is calculated as:

`log1p(last_13_week_spend) - log1p(first_13_week_spend)`

This transformed feature measures the difference between the two comparison periods. It is not an ordinary percentage growth rate.

For stores with zero starting-period spend, log spend change is treated as missing for clustering rather than being interpreted as a conventional growth measure.

7.5 Product-category mix

The product-category features represent the share of store spend associated with selected product categories.

The clustering pipeline selects the 15 product categories with the highest total spend across stores and creates a separate category-share feature for each selected category.

7.6 Historical and contextual features

The enriched store data also contains historical activity and contextual fields, including active weeks, store format, store region, and the zero-starting-spend indicator.

These fields support data-quality checks and segment interpretation. Not all of them are numerical inputs to the clustering model.

In the current clustering implementation, the numerical feature matrix includes:

- Log-transformed average weekly spend.
- Relative spend volatility.
- Log spend change.
- Selected product-category share features.

7.7 Missing-value treatment and scaling

Infinite values are replaced with missing values. Missing numerical feature values are imputed using the median, and the resulting feature matrix is standardized using `StandardScaler`.

The same feature preparation must be used when reproducing assignments or comparing models.

8. Models Evaluated

Two unsupervised learning algorithms were evaluated.

8.1 K-Means

K-Means partitions stores into a specified number of clusters by minimizing within-cluster squared distances to cluster centroids.

The current clustering configuration uses:

- Random state: `42`
- Number of initializations: `20`
- Evaluated cluster counts: `2` through `8`

8.2 Agglomerative Clustering

Agglomerative Clustering creates a hierarchy of groups by progressively merging observations. The current implementation uses Ward linkage.

Cluster counts from two through eight were evaluated.

8.3 Evaluation metrics

The model comparison uses:

- Silhouette score: measures cluster cohesion and separation; higher values are generally preferred.
- Davies–Bouldin score: measures similarity between clusters; lower values are generally preferred.
- Calinski–Harabasz score: compares between-cluster dispersion with within-cluster dispersion; higher values are generally preferred.
- Cluster sizes: identify very small or highly imbalanced groups.
- Interpretability: assesses whether the resulting groups provide meaningful distinctions for business investigation.

No single metric proves that a clustering solution is universally best. The metrics are considered together with the segment profiles and intended business use.

9. Model Comparison and Selection

The reported initial model comparison produced the following results:

| Algorithm | Clusters | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|---|---:|---:|---:|---:|
| K-Means | 2 | 0.3248 | 1.8196 | 155.6441 |
| K-Means | 3 | 0.3164 | Verify from generated report | 114.4322 |
| Agglomerative | 3 | 0.2940 | 1.7446 | 107.2781 |
| Agglomerative | 2 | 0.2910 | 2.0147 | 144.2133 |

The three-cluster K-Means Davies–Bouldin value should be copied directly from `reports/clustering_model_comparison.csv` before submission.

K-Means with two clusters achieved the highest silhouette score among the listed candidates and the highest Calinski–Harabasz score in this comparison.

The three-cluster K-Means solution had a lower reported silhouette score, but the two-cluster model did not outperform every alternative on every metric.

K-Means with two clusters was selected as the primary solution because it provides a strong silhouette result and produces two groups that can be interpreted in terms of spend scale and variability.

The three-cluster solution is retained as a comparison because alternative cluster counts may capture additional differences between stores.

10. Final Store Segments

The selected two-cluster solution assigns all 761 stores to two segments.

10.1 Segment 1: Smaller-Spend Stores

| Characteristic | Reported value |
|---|---:|
| Number of stores | 139 |
| Share of analyzed stores | 18.27% |
| Median store-level average weekly spend | 142.17 |
| Mean store-level average weekly spend | 171.89 |
| Median weekly spend volatility | 45.06 |
| Median relative spend volatility | 0.33 |
| Median active weeks | 117 |
| Stores with zero starting-period spend | 0 |

This segment contains stores with lower typical weekly spend and higher relative spend variability.

Potential business actions include:

- Reviewing the assortment of major product categories.
- Comparing category mix with similar stores in the same format and region.
- Identifying possible product gaps for further investigation.
- Testing targeted assortment changes or promotions in a small group of comparable stores.
- Tracking weekly spend, unit sales, category mix, and variability before and after any intervention.

10.2 Segment 2: Larger-Spend Stores

| Characteristic | Reported value |
|---|---:|
| Number of stores | 622 |
| Share of analyzed stores | 81.73% |
| Median store-level average weekly spend | 751.51 |
| Mean store-level average weekly spend | 786.31 |
| Median weekly spend volatility | 160.45 |
| Median relative spend volatility | 0.21 |
| Median active weeks | 117 |
| Stores with zero starting-period spend | 10 |

This segment contains stores with higher typical weekly spend and lower relative spend variability, although their absolute weekly spend volatility is higher.

Potential business actions include:

- Reviewing high-contribution product categories.
- Investigating category-specific assortment opportunities.
- Examining replenishment requirements with inventory and availability data.
- Monitoring weekly spend, units sold, category mix, and operational service levels.
- Comparing results across store formats and regions before applying changes.

10.3 Store-format composition

The reported format analysis indicates that approximately 66.9% of smaller-spend stores are SS format, while approximately 52.4% of larger-spend stores are LS format.

This association suggests that store format should be considered when interpreting the segments. It does not establish that store format causes a particular spend level.

The labels “Smaller-Spend Stores” and “Larger-Spend Stores” are descriptive labels for this dataset, not universal classifications.

The reported segment statistics should be validated against the generated final assignment and profile files before submission.

11. Stability and Sensitivity Analysis

The project evaluates clustering across multiple random seeds and with and without the store-scale feature.

The reported results are:

| Configuration | Clusters | Mean silhouette | Mean adjusted Rand index |
|---|---:|---:|---:|
| With store scale | 2 | 0.3245 | 0.9984 |
| With store scale | 3 | 0.3167 | 0.9973 |
| Without store scale | 2 | 0.3122 | 1.0000 |
| Without store scale | 3 | 0.3131 | 0.9916 |

These values should be checked against the generated stability and sensitivity reports before submission.

The adjusted Rand index (ARI) measures agreement between clustering assignments. Values closer to 1 indicate stronger agreement, and 1 indicates identical partitions.

The reported results suggest that the cluster assignments were highly consistent across the tested random seeds. Removing the store-scale feature slightly reduced the mean silhouette score for the two-cluster solution, while assignment agreement remained high.

A separate reproducibility check compared a freshly calculated two-cluster K-Means model with the saved `kmeans_2_assignments.csv` file. All 761 stores were matched, and the adjusted Rand index was **1.0**.

This confirms that the fresh run reproduced the saved partition under the tested feature preparation and model configuration.

However, this check does not independently verify every stage of the complete workflow. Automatic generation of both two-cluster and three-cluster assignment files must also be incorporated into and tested through the clustering pipeline.

These tests assess random-seed and selected feature-set sensitivity. They do not establish robustness under data resampling, new time periods, or changes in the retail environment.

12. Business Recommendations

The segments provide a starting point for proposing targeted business actions. Recommendations should be tested rather than assumed to be effective.

12.1 Recommendations for smaller-spend stores

- Review the assortment of major product categories and investigate possible gaps.
- Compare stores with similar formats and regional characteristics.
- Pilot targeted assortment changes or promotions in a small group of comparable stores.
- Track spend, unit sales, category mix, and variability before and after the intervention.
- Compare outcomes against a suitable control group before expanding the intervention.

12.2 Recommendations for larger-spend stores

- Review high-contribution product categories and assortment opportunities.
- Examine replenishment requirements using verified inventory and availability information.
- Test assortment improvements while monitoring spend, units sold, and operational service levels.
- Investigate differences between store formats and regions.
- Avoid treating higher spend as proof of higher profitability.

12.3 Recommendations for both segments

- Use store format and region to refine segment interpretation.
- Define measurable success criteria before each experiment.
- Use comparable stores or control groups to evaluate the impact of interventions.
- Reassess the segments when more recent transaction data becomes available.
- Monitor changes in store behavior over time.

These are proposed actions for investigation, not proven outcomes. The current data does not directly measure promotion response, stockouts, profit margins, or causal effects.

13. Limitations

13.1 Business-feature availability

The dataset includes transaction spend, quantity, product categories, store format, and region. Basket characteristics such as price sensitivity, basket type, and dominant shopping mission are described in the project analysis but should only be treated as available features if they are supported by the actual source data.

The following business features have not been verified in the available data:

- Promotions and discounts: Explicit promotion identifiers and reliable promotion-effectiveness measures have not been verified.
- Pricing: Transaction spend is available, but comparable unit selling prices and discount amounts have not been validated.
- Inventory and stockouts: Inventory-on-hand and stockout records have not been verified.
- Distribution and product availability: Explicit store-product distribution and availability records have not been verified.
- Profitability: Product cost and margin fields have not been verified.

Consequently, sales spikes cannot automatically be attributed to promotions, low sales cannot automatically be interpreted as low demand, and the segments should be interpreted as sales-behavior segments rather than profitability segments.

13.2 Growth interpretation

Growth estimates depend on the selected first and last 13-week comparison windows. Low starting-period spend can create very large relative changes, while zero starting-period spend makes conventional percentage growth undefined.

The available transaction data alone cannot establish why a store experienced an extreme change. Store opening dates, store activity records, and operational data would help improve interpretation.

13.3 Modeling limitations

Clustering is an unsupervised method, so there are no predefined ground-truth labels against which to measure classification accuracy.

The model-selection metrics evaluate different aspects of cluster structure and do not guarantee business usefulness. Feature selection, imputation, scaling, and the number of clusters influence the resulting groups.

Although random-seed and selected feature-set sensitivity were examined, further validation using resampling, later time periods, and additional operational variables would strengthen confidence in the segmentation.

13.4 Business interpretation

The analysis is descriptive. It does not establish that a particular store characteristic causes higher or lower spend, nor does it prove that a recommended intervention will improve business performance.

Verified pricing, promotion, inventory, availability, and margin information would be useful additions to future analyses.

14. Project Structure

The project structure should reflect the actual files in the repository. Based on the previously inspected project, the expected layout is:

```text
FreshBasket_Store_Segmentation/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── analyze_seasonality.py
│   ├── build_final_features.py
│   ├── build_store_features.py
│   ├── create_final_deliverables.py
│   ├── data_preprocessing.py
│   ├── data_quality_eda.py
│   ├── inspect_cluster_profiles.py
│   ├── inspect_features.py
│   ├── review_growth_outliers.py
│   ├── run_clustering.py
│   └── test_cluster_stability.py
├
├── reports/
├── README.md
└── requirements.txt
```

The `notebooks/` directory may also be present, but it should only be included in the tree if it exists and contains relevant project files.

15. Key Outputs

The following files are part of the intended project outputs. Confirm their existence before listing them as completed deliverables.

15.1 Processed data

- `data/processed/store_week_product_sales.csv`
- `data/processed/store_week_sales.csv`
- `data/processed/store_category_mix.csv`
- `data/processed/store_features_initial.csv`
- `data/processed/store_features_enriched.csv`
- `data/processed/store_cluster_assignments.csv`
- `data/processed/kmeans_2_assignments.csv`
- `data/processed/kmeans_3_assignments.csv`
- `data/processed/final_store_segment_assignments.csv`

15.2 Clustering and segment reports

- `reports/clustering_model_comparison.csv`
- `reports/cluster_profiles.csv`
- `reports/cluster_store_format_mix.csv`
- `reports/cluster_store_region_mix.csv`
- `reports/cluster_category_mix.csv`
- `reports/cluster_stability_sensitivity.csv`
- `reports/cluster_sensitivity_summary.csv`
- `reports/final_segment_profile.csv`
- `reports/final_segment_format_mix.csv`
- `reports/final_segment_region_mix.csv`
- `reports/final_segment_category_profile.csv`
- `reports/segment_recommendation_matrix.csv`

15.3 Data-quality and supporting reports

- `reports/growth_feature_audit.csv`
- `reports/growth_period_review.csv`
- `reports/seasonality_weekly_summary.csv`
- `reports/seasonality_monthly_summary.csv`
- `reports/seasonality_week_of_year_summary.csv`
- `reports/seasonality_year_summary.csv`

The exact list depends on which scripts have been executed successfully. The README should not imply that an output exists if it has not been generated.

16. Running the Project

16.1 Requirements

- Python 3.
- pandas.
- NumPy.
- scikit-learn.
- Matplotlib.
- Seaborn.

Use the project's virtual environment and install the dependencies listed in `requirements.txt`.

From PowerShell, activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If dependencies are listed in `requirements.txt`, install them using:

```powershell
python -m pip install -r requirements.txt
```

16.2 Execution order

Run scripts from the project root so relative paths resolve correctly.

The general dependency order is:

1. Inspect the input files and schemas.
2. Preprocess raw transaction data.
3. Build store-level features and aggregated datasets.
4. Generate data-quality and EDA reports.
5. Review growth outliers.
6. Build enriched store features.
7. Analyze seasonality.
8. Run clustering model comparison.
9. Inspect cluster profiles and test stability.
10. Generate final assignments, segment profiles, and recommendations.
11. Validate the generated outputs against this README.

The exact commands should be confirmed against each script's current inputs and outputs before running the complete workflow. Some preprocessing stages read the large raw dataset and may take considerable time.

Do not rerun expensive raw-data processing unnecessarily if the processed inputs are already available and valid.

16.3 Reproducibility

Reproducing the results requires consistent input data, feature preparation, imputation, scaling, and model parameters.

The current K-Means configuration uses `random_state=42` and `n_init=20`. Any change to these parameters, the selected product categories, or the numerical feature matrix may change the cluster assignments.

The complete workflow should automatically generate the required cluster-assignment files and reports. A separate two-cluster reproducibility check achieved an adjusted Rand index of 1.0 against the saved assignments, but end-to-end reproducibility should only be claimed after the complete pipeline has been tested.

17. Future Enhancements

Potential future improvements include:

- Incorporating verified product prices, discounts, promotions, and margins.
- Adding inventory, stockout, and product-availability information.
- Evaluating segmentation across later time periods.
- Testing robustness under data resampling.
- Comparing alternative feature sets and clustering methods.
- Refining recommendations by store format and region.
- Conducting controlled experiments to measure the impact of segment-specific actions.
- Automating the complete pipeline, validation checks, and report generation.

These improvements would help bridge the gap between descriptive clustering and operational decision-making.

18. Conclusion

This project establishes a baseline for segmenting 761 FMCG retail stores using transaction-derived features and unsupervised machine learning.

K-Means with two clusters achieved the highest silhouette score among the reported candidates and showed high consistency across the tested random seeds. The resulting segments distinguish smaller-spend stores with higher relative spend variability from larger-spend stores with lower relative spend variability.

The segments provide a foundation for investigating targeted assortment, replenishment, and store-level business experiments. A separate reproducibility check achieved an adjusted Rand index of 1.0 between a fresh two-cluster run and the saved assignments.

Complete pipeline reproducibility should be confirmed after automatic generation of all required cluster-assignment files has been incorporated and tested.

Future work should incorporate verified pricing, promotion, inventory, and margin data and evaluate whether segment-specific actions improve measurable business outcomes.
