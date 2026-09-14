# Data Preprocessing & Exploratory Analysis Report

## Dataset Overview
- **Source File:** `train-test.csv`
- **Total Rows (Observations):** 48,000
- **Total Columns (Features):** 14
- **Duplicate Rows:** 0 (Clean dataset in terms of exact row duplicates)

---

## Feature Schema & Data Types

| Column Name | Data Type | Description / Observations |
| :--- | :--- | :--- |
| `load_id` | `object` (String) | Unique identifier for each freight load (e.g., `TR-000001`). |
| `pickup` | `object` (String) | Origin city / location name. |
| `delivery` | `object` (String) | Destination city / location name. |
| `pickup_lat` | `float64` | Latitude coordinate of the pickup location. |
| `pickup_lon` | `float64` | Longitude coordinate of the pickup location. |
| `delivery_lat` | `float64` | Latitude coordinate of the delivery location. |
| `delivery_lon` | `float64` | Longitude coordinate of the delivery location. |
| `distance` | `float64` | Distance of the route (miles/kilometers). |
| `equipment` | `object` (String) | Type of truck body required to haul the freight.|
| `weight` | `float64` | Weight of the freight cargo. |
| `date` | `object` (String) | Date of the transaction/load posting (needs parsing to datetime). |
| `market_index` | `float64` | Market index indicator rate. |
| `quote_signal` | `float64` | Signal score or pricing quote metric. |
| `posted_rate` | `float64` | The posted freight rate (Target variable or key metric). |

---

## Missing Value Audit & Handling Strategy

| Column | Missing Count | Percentage | Current Strategy (Selected) |
| :--- | :--- | :--- | :--- |
| `weight` | 300 | ~0.625% | **Drop rows** (Low volume impact) |
| `market_index` | 374 | ~0.779% | **Drop rows** (Low volume impact) |
| *All other columns* | 0 | 0.00% | No missing value treatment required. |

### Rationale for Dropping
- **Minimal Data Loss:** With 48,000 total records, dropping rows containing missing values in `weight` and `market_index` removes roughly 1.4% of the dataset combined. This has a negligible impact on statistical power and avoids introducing artificial variance or imputation bias into a baseline model.

---

## Production Considerations & Best Practices

While dropping rows is convenient for offline exploratory analysis and initial model training, transitioning to a **production environment** introduces critical constraints that change how missing data must be handled:

1. **Inference-Time Realities:**
   - In production, incoming live requests (e.g., a customer requesting a rate quote) will occasionally lack optional or poorly logged fields (like weight or market index). 
   - **The Problem:** You cannot simply "drop" a live customer request because a field is missing. Rejecting live traffic damages user experience and conversion rates.

2. **Data Leakage & Pipeline Consistency:**
   - If a model is trained on a strictly cleaned (dropped) subset, but production endpoints receive data with nulls, the inference script will crash unless explicit handling is built-in.
   - **The Solution:** Production pipelines require **explicit imputation fallback logic** (e.g., fallback medians derived from training data, or default category defaults) or robust ML-based imputation steps integrated directly into feature engineering pipelines (such as Scikit-Learn `Pipeline` and `SimpleImputer`).

3. **Potential Hidden Bias:**
   - Even at <1.4%, if missingness is non-random (e.g., specific high-capacity carriers failing to report weights), dropping rows permanently removes those specific edge cases from the training distribution, potentially blind-spotting the model in production.

---

## Data Quality & Type Consistency Check
- **Mixed Python Types Test:** A type consistency scan was executed across all columns (`df[col].dropna().map(type)`). 
- **Result:** **0 unexpected type discrepancies** were found in any column. Every feature strictly conforms to a single uniform Python data type.

---

## Key Observations & Next Steps

1. **Clean Structural Integrity:** The dataset contains 48,000 records with zero duplicate rows and no type-mixing anomalies.
2. **Execute Cleaning Step:** Drop the rows containing null values in `weight` and `market_index` to streamline the exploratory phase.
3. **Temporal Processing Required:** Convert the `date` column using `pd.to_datetime()` to extract seasonal/temporal features.
4. **Prepare for Production:** Plan ahead for robust imputation handling if this model is slated for deployment where real-time missing values cannot be dropped.
"""

with open("preprocessing.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)

print("preprocessing.md successfully updated!")