
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Load the Gallup + O*NET dataset

project_folder = Path(__file__).resolve().parent.parent

data_path = (
    project_folder
    / "notebooks"
    / "gallup_eda"
    / "output"
    / "gallup_fsu_pre_model.csv"
)

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("\nJob satisfaction distribution:")
print(df["satisfied"].value_counts())


# 2. Select background features

# Numeric features
numeric_features = ["Q20", "Q51"]

# Categorical features
categorical_features = [
    "Q8", "Q9", "Q14", "Q16", "Q17", "Q40"
]

X = df[numeric_features + categorical_features].copy()
y = df["satisfied"]


# 3. Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAIN/TEST SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# 4. Preprocess numeric and categorical features

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# 5. Train logistic regression

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)


# 6. Evaluate on the test set

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nBASELINE LOGISTIC REGRESSION RESULTS")
print("ROC-AUC:", round(auc, 4))
print("Balanced accuracy:", round(balanced_acc, 4))
print("F1 score:", round(f1, 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))


# 7. Calculate 95% bootstrap confidence interval for ROC-AUC

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

    # ROC-AUC requires both outcome classes
    if len(np.unique(sampled_y)) < 2:
        continue

    bootstrap_aucs.append(
        roc_auc_score(sampled_y, sampled_prob)
    )

lower, upper = np.percentile(bootstrap_aucs, [2.5, 97.5])

print("\nROC-AUC 95% BOOTSTRAP CONFIDENCE INTERVAL")
print(f"AUC: {auc:.4f}")
print(f"95% CI: ({lower:.4f}, {upper:.4f})")


