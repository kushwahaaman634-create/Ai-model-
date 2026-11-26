# Phishing URL Detection with Random Forest

A machine learning project that classifies URLs as phishing or legitimate using hand-crafted features extracted from the URL and its page. The model is a `RandomForestClassifier` trained on a tabular dataset of 47 numeric features.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model](#model)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Limitations and Next Steps](#limitations-and-next-steps)

---

## Overview

Phishing sites imitate legitimate ones to steal credentials and personal data. This project frames detection as binary classification: given a set of features describing a URL and its page, predict whether it is phishing (label `1`) or legitimate (label `0`).

The workflow covers:

1. Cleaning the raw dataset
2. Analysing feature correlations
3. Training a Random Forest classifier
4. Evaluating it on a held-out test set and with cross-validation

## Dataset

| Property | Value |
|---|---|
| Original shape | 10,000 rows × 50 columns |
| Cleaned shape | 9,581 rows × 48 columns (47 features + target) |
| Target column | `CLASS_LABEL` |
| Class balance | `0`: 4,996 · `1`: 4,585 |
| Missing values | None |
| Duplicates | 419 removed after dropping the ID column |

**Features.** All 47 predictors are numeric. They cover:

- **URL structure:** `UrlLength`, `PathLength`, `PathLevel`, `HostnameLength`, `NumDots`, `NumDash`, `NumUnderscore`, `NumPercent`, `NumQueryComponents`, `NumAmpersand`, `NumHash`, `NumNumericChars`, `AtSymbol`, `TildeSymbol`, `DoubleSlashInPath`
- **Domain and host:** `SubdomainLevel`, `DomainInSubdomains`, `DomainInPaths`, `NumDashInHostname`, `IpAddress`, `NoHttps`, `FrequentDomainNameMismatch`, `EmbeddedBrandName`
- **Page content and links:** `PctExtHyperlinks`, `PctExtResourceUrls`, `PctNullSelfRedirectHyperlinks`, `PctExtNullSelfRedirectHyperlinksRT`, `ExtFavicon`, `InsecureForms`, `RelativeFormAction`, `ExtFormAction`, `AbnormalFormAction`, `SubmitInfoToEmail`, `IframeOrFrame`, `PopUpWindow`, `RightClickDisabled`, `FakeLinkInStatusBar`, `MissingTitle`, `ImagesOnlyInForm`
- **Text signals:** `NumSensitiveWords`, `RandomString`

**Removed during cleaning:** `id` (identifier, not predictive) and `HttpsInHostname` (constant).

## Pipeline

```
dataset.csv
   │
   ├─ Drop identifier and constant columns
   ├─ Remove duplicate records
   ├─ Check for missing values and feature correlations (|r| ≥ 0.90)
   │
   ▼
cleaned_dataset_random_forest.csv
   │
   ├─ Stratified train/test split (80/20, random_state=42)
   ├─ Fit RandomForestClassifier (300 trees)
   │
   ▼
random_forest_phishing_model.joblib
   │
   └─ Evaluate: holdout metrics, confusion matrix, ROC-AUC,
                5-fold stratified cross-validation, feature importances
```

**Preprocessing decisions:**

- **Scaling:** Not applied. Random Forests are invariant to monotonic feature scaling.
- **Encoding:** Not needed. All retained features are numeric.
- **Imputation:** Not needed. There are no missing values.
- **Correlation:** No predictor pairs have |r| ≥ 0.90, so no features were dropped for collinearity.

## Model

**Algorithm:** `RandomForestClassifier` from scikit-learn

| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `random_state` | 42 |
| `max_features` | `sqrt` (default) |
| `max_depth` | `None` (default) |
| `min_samples_leaf` | 1 (default) |
| `class_weight` | `None` |

These are default-style settings and have not been tuned.

## Results

### Holdout test set (n = 1,917)

| Metric | Score |
|---|---|
| Accuracy | 0.9875 |
| Precision | 0.9859 |
| Recall | 0.9880 |
| F1-score | 0.9869 |
| ROC-AUC | 0.9993 |

**Confusion matrix** (rows = true, columns = predicted):

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **True 0** | 987 | 13 |
| **True 1** | 11 | 906 |

### 5-fold stratified cross-validation

| Metric | Mean ± Std |
|---|---|
| Accuracy | 0.9819 ± 0.0021 |
| Precision | 0.9804 ± 0.0032 |
| Recall | 0.9819 ± 0.0026 |
| F1-score | 0.9811 ± 0.0022 |
| ROC-AUC | 0.9984 ± 0.0003 |

The cross-validation mean is the more reliable estimate of expected performance. The single holdout split is about half a point higher.

### Top feature importances

1. `PctExtHyperlinks` (0.187)
2. `PctExtNullSelfRedirectHyperlinksRT` (0.162)
3. `PctExtResourceUrls` (0.090)
4. `FrequentDomainNameMismatch` (0.078)
5. `PctNullSelfRedirectHyperlinks` (0.065)

Hyperlink and external-resource features dominate the model's decisions.

A full write-up is in `random_forest_model_evaluation.txt`.

## Project Structure

```
.
├── README.md                              # This file
├── preprocessing_random_forest.py         # Cleaning, deduplication, correlation analysis
├── preprocessing_report.txt               # Summary of preprocessing decisions
├── cleaned_dataset_random_forest.csv      # Cleaned dataset (9,581 × 48)
├── feature_correlation_heatmap.png        # Pearson correlation heatmap of predictors
├── random_forest_phishing_model.joblib    # Trained model
├── random_forest_evaluation_report.txt    # Holdout evaluation output
└── random_forest_model_evaluation.txt     # Full evaluation report with cross-validation
```

## Getting Started

### Requirements

- Python 3.9 or newer (the model was developed with Python 3.12)
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib (for the heatmap)

Install with:

```bash
pip install pandas numpy scikit-learn joblib matplotlib
```

For reproducibility, pin the scikit-learn version used in training. The saved model was created with scikit-learn 1.8.0.

### Run preprocessing

Place the raw file as `dataset.csv` in the project root, then run:

```bash
python preprocessing_random_forest.py
```

This writes `cleaned_dataset_random_forest.csv` and prints the correlation analysis.

### Load the model and predict

```python
import joblib
import pandas as pd

model = joblib.load("random_forest_phishing_model.joblib")
df = pd.read_csv("cleaned_dataset_random_forest.csv")

X = df.drop(columns=["CLASS_LABEL"])
# Columns must match model.feature_names_in_ in the same order
X = X[model.feature_names_in_]

predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]
```

> Confirm the meaning of label `1` (phishing) against the dataset documentation before relying on predictions.
