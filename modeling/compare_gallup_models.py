
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# STEP 10: Recreate Model 1 predictions

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Load the merged Gallup + O*NET dataset
data_path = (
    project_folder
    / "notebooks"
    / "gallup_eda"
    / "output"
    / "gallup_fsu_pre_model.csv"
)

df = pd.read_csv(data_path)

# Use the same background features as Model 1
numeric_features = ["Q20", "Q51"]

categorical_features = [
    "Q8", "Q9", "Q14", "Q16", "Q17", "Q40"
]

X = df[numeric_features + categorical_features].copy()
y = df["satisfied"]

# Recreate the original train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preprocess numeric features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocess categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Train the same baseline logistic regression model
model1 = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

model1.fit(X_train, y_train)

# Predict satisfaction probabilities
model1_prob = model1.predict_proba(X_test)[:, 1]

# Calculate Model 1 ROC-AUC
model1_auc = roc_auc_score(y_test, model1_prob)

print("\nMODEL 1 RECREATED")
print("Test respondents:", len(X_test))
print("ROC-AUC:", round(model1_auc, 4))


# STEP 11: Compare Model 1 and Model 2

import numpy as np

# Load Model 2's saved predictions
predictions_path = (
    project_folder
    / "modeling"
    / "output"
    / "gallup_onet_predictions.csv"
)

model2_predictions = pd.read_csv(predictions_path)

# Create a DataFrame with Model 1 predictions
model1_predictions = pd.DataFrame({
    "row_index": X_test.index,
    "actual": y_test.to_numpy(),
    "model1_probability": model1_prob
})

# Match both models by respondent row index
comparison = model1_predictions.merge(
    model2_predictions,
    on="row_index",
    suffixes=("_model1", "_model2"),
    validate="one_to_one"
)

# Check that every test respondent is included
assert len(comparison) == len(X_test)

# Check that the actual outcomes match
assert (
    comparison["actual_model1"]
    == comparison["actual_model2"]
).all()

# Calculate ROC-AUC for both models
actual = comparison["actual_model1"].to_numpy()
prob1 = comparison["model1_probability"].to_numpy()
prob2 = comparison["model2_probability"].to_numpy()

auc1 = roc_auc_score(actual, prob1)
auc2 = roc_auc_score(actual, prob2)

auc_difference = auc2 - auc1

print("\nMODEL COMPARISON")
print("Model 1 ROC-AUC:", round(auc1, 4))
print("Model 2 ROC-AUC:", round(auc2, 4))
print("AUC difference:", round(auc_difference, 4))


# STEP 12: Paired bootstrap confidence interval

rng = np.random.default_rng(42)
bootstrap_differences = []

for i in range(1000):
    # Use the same sampled respondents for both models
    indices = rng.integers(0, len(actual), len(actual))

    sampled_y = actual[indices]
    sampled_prob1 = prob1[indices]
    sampled_prob2 = prob2[indices]

    # ROC-AUC requires both outcome classes
    if len(np.unique(sampled_y)) < 2:
        continue

    sample_auc1 = roc_auc_score(sampled_y, sampled_prob1)
    sample_auc2 = roc_auc_score(sampled_y, sampled_prob2)

    bootstrap_differences.append(sample_auc2 - sample_auc1)

lower, upper = np.percentile(
    bootstrap_differences,
    [2.5, 97.5]
)

print("\nPAIRED BOOTSTRAP RESULTS")
print(f"AUC difference: {auc_difference:.4f}")
print(f"95% CI: ({lower:.4f}, {upper:.4f})")

if lower > 0:
    print("Model 2 shows evidence of better ROC-AUC.")
elif upper < 0:
    print("Model 1 shows evidence of better ROC-AUC.")
else:
    print("No statistically clear AUC improvement detected.")

