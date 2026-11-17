
from pathlib import Path
import pandas as pd

# Find the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Locate the merged Gallup + O*NET dataset
data_path = (
    project_folder
    / "notebooks"
    / "gallup_eda"
    / "output"
    / "gallup_fsu_pre_model.csv"
)

# Load the dataset
df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)

# Find all O*NET feature columns
onet_columns = [
    col for col in df.columns
    if col.startswith("onet_")
]

print("\nONET FEATURE SUMMARY")
print("Number of O*NET columns:", len(onet_columns))

print("\nFirst 30 O*NET columns:")
for col in onet_columns[:30]:
    print(col)

print("\nO*NET missing values:")
print(df[onet_columns].isna().sum().describe())

print("\nMatching status:")
print(df["match_status"].value_counts(dropna=False))


# STEP 3: Inspect O*NET feature types

print("\nO*NET DATA TYPES")
print(df[onet_columns].dtypes.value_counts())

print("\nO*NET FEATURE GROUPS")

# Group columns by their prefixes
feature_groups = {}

for col in onet_columns:
    # Remove "onet_" and take the letters before the first number
    name = col.replace("onet_", "")
    prefix = ""

    for char in name:
        if char.isalpha():
            prefix += char
        else:
            break

    if prefix not in feature_groups:
        feature_groups[prefix] = []

    feature_groups[prefix].append(col)

for prefix, columns in sorted(feature_groups.items()):
    print(f"{prefix}: {len(columns)} features")
    print("  Example:", columns[0])

print("\nSAMPLE O*NET VALUES")
print(df[onet_columns[:5]].head())


# STEP 4: Prepare background + O*NET features

from sklearn.model_selection import train_test_split

# Same background features used in Model 1
numeric_features = ["Q20", "Q51"]

categorical_features = [
    "Q8", "Q9", "Q14", "Q16", "Q17", "Q40"
]

# Combine numeric background features with O*NET features
all_numeric_features = numeric_features + onet_columns

# X contains predictors, y contains job satisfaction
X = df[all_numeric_features + categorical_features].copy()
y = df["satisfied"]

# Use the same split settings as the baseline
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nMODEL 2 TRAIN/TEST SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nMODEL 2 FEATURE COUNTS")
print("Numeric features:", len(all_numeric_features))
print("Categorical features:", len(categorical_features))
print("Total original features:", X.shape[1])

print("\nMissing O*NET values in training:")
print(X_train[onet_columns].isna().sum().sum())

print("\nMissing O*NET values in testing:")
print(X_test[onet_columns].isna().sum().sum())



# STEP 5: Preprocess background and O*NET features

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Prepare numeric features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Prepare categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine numeric and categorical preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, all_numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("\nMODEL 2 PREPROCESSING READY")
print("Numeric features:", len(all_numeric_features))
print("Categorical features:", len(categorical_features))



# STEP 6: Train and evaluate Model 2

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Combine preprocessing and logistic regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# Train the model using training data only
model.fit(X_train, y_train)

# Predict job satisfaction on the test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
auc = roc_auc_score(y_test, y_prob)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nMODEL 2 LOGISTIC REGRESSION RESULTS")
print("ROC-AUC:", round(auc, 4))
print("Balanced accuracy:", round(balanced_acc, 4))
print("F1 score:", round(f1, 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))



# STEP 7: Calculate 95% bootstrap confidence interval

import numpy as np

rng = np.random.default_rng(42)
bootstrap_aucs = []

y_test_array = y_test.to_numpy()

for i in range(1000):
    # Sample test respondents with replacement
    indices = rng.integers(
        0, len(y_test_array), len(y_test_array)
    )

    sampled_y = y_test_array[indices]
    sampled_prob = y_prob[indices]

    # AUC requires both satisfaction classes
    if len(np.unique(sampled_y)) < 2:
        continue

    bootstrap_aucs.append(
        roc_auc_score(sampled_y, sampled_prob)
    )

lower, upper = np.percentile(bootstrap_aucs, [2.5, 97.5])

print("\nMODEL 2 ROC-AUC 95% CONFIDENCE INTERVAL")
print(f"AUC: {auc:.4f}")
print(f"95% CI: ({lower:.4f}, {upper:.4f})")



# STEP 8: Save Model 2 test predictions for comparison

output_folder = project_folder / "modeling" / "output"
output_folder.mkdir(parents=True, exist_ok=True)

predictions = pd.DataFrame({
    "row_index": X_test.index,
    "actual": y_test.to_numpy(),
    "model2_probability": y_prob
})

output_path = output_folder / "gallup_onet_predictions.csv"
predictions.to_csv(output_path, index=False)

print("\nMODEL 2 PREDICTIONS SAVED")
print("File:", output_path)
print("Rows:", len(predictions))
