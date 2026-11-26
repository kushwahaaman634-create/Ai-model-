import pandas as pd

# Load original dataset
df = pd.read_csv("dataset.csv")

print("Original shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Exact duplicate rows:", int(df.duplicated().sum()))

# Drop identifier (not a predictive feature)
df = df.drop(columns=["id"], errors="ignore")

# Remove constant columns
constant_cols = df.columns[df.nunique(dropna=False) <= 1].tolist()
print("Constant columns removed:", constant_cols)
df = df.drop(columns=constant_cols)

# This dataset has no missing values, so no imputation is required.
# All retained features are numeric, so categorical encoding is unnecessary.

# Remove duplicate feature/target records
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print("Duplicate records removed:", before - len(df))

# Save cleaned dataset
df.to_csv("cleaned_dataset_random_forest.csv", index=False)
print("Cleaned shape:", df.shape)

# Correlation analysis (predictors only)
target = "CLASS_LABEL"
X = df.drop(columns=[target])
corr = X.corr(numeric_only=True)

high_corr = []
columns = list(corr.columns)
for i, col1 in enumerate(columns):
    for col2 in columns[i + 1:]:
        value = corr.loc[col1, col2]
        if abs(value) >= 0.90:
            high_corr.append((col1, col2, value))

print("Feature pairs with |correlation| >= 0.90:")
for item in high_corr:
    print(item)

# Target correlations are for exploratory analysis only.
print("\\nTop correlations with target:")
print(df.corr(numeric_only=True)[target].drop(target).abs().sort_values(
    ascending=False
).head(15))

# Recommended next step:
# Use a stratified train/test split. If repeated source entities exist,
# use group-aware splitting to prevent entity leakage.
