# IEEE-CIS Fraud Detection — Module 3

## Dataset
- Source: IEEE-CIS Fraud Detection — Vesta Corporation / Kaggle Competition
- Files: train_transaction.csv + train_identity.csv
- Join: left join on TransactionID → 590,540 rows

## Key Insights from EDA

1. **Highly imbalanced dataset** — 3.4% fraud vs 96.6% non-fraud.
   Standard accuracy is not a valid metric. F1 score and precision-recall
   tradeoff will guide model evaluation.

2. **Temporal split is mandatory** — TransactionDT spans 182 days.
   A random split would introduce data leakage. Split strategy:
   60/20/20 chronological (109/36/36 days).

3. **V features dominate missing values** — groups V138-V339 have 76-86%
   nulls. Features above 80% missing will be dropped. M features encode
   NaN as a meaningful category, not missing data.

## Infrastructure Notes

| Dataset | Rows | Columns | Memory | Load Time |
|---------|------|---------|-------|-----------|
| train_transaction | 590,540 | 394     | 2062.1 MB | 18.74s    |
| train_identity | 144,233 | 41      | 143.1 MB | 0.44s     |
| merged | 590,540 | 437 | 2519.6 MB | —         |

**Observations:**
- The merged dataset requires ~2.5GB of RAM. A production feature engineering
  pipeline would need at minimum 8GB available, with 16GB recommended for
  training with XGBoost (initially chosen model).
- Load time of 18.74s for the transaction file suggests that in a streaming
  inference scenario, features would need to be pre-computed and stored in a
  feature store rather than recomputed on demand.
- The identity file is 14x smaller than the transaction file, confirming that
  ~75% of transactions have no associated identity data — reinforcing the
  decision to create the `has_identity` binary feature.

## Project Structure

```
module3-fraud-detection/
├── data/
│   ├── train_identity.csv              # Identity data set
│   ├── train_transaction.csv           # Transaction data set
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb          # Feature engineering & data preparation
│   ├── 03_training.ipynb               # Model training & evaluation
│   └── 04_model_comparison.ipynb       # XGBoost vs Neural Network comparison
├── src/
│   ├── preprocessing.py                # Reusable preprocessing pipeline
│   └── serving/
│       └── main.py                     # FastAPI inference endpoint
├── adrs/
│   ├── 001_join_strategy.md
│   ├── 002_chronological_split.md
│   └── 003_model_selection.md
├── docs/
│   └── cost_analysis.md
└── README.md
```