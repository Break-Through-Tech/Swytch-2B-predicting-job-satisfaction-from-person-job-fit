
# STEP 1: Load the Gallup + O*NET dataset

from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent

data_path = (
    project_folder
    / "notebooks"
    / "gallup_eda"
    / "output"
    / "gallup_fsu_pre_model.csv"
)

df = pd.read_csv(data_path)

# Identify O*NET features
onet_columns = [
    col for col in df.columns
    if col.startswith("onet_")
]

print("\nTUNED MODEL DATASET")
print("Rows:", len(df))
print("O*NET features:", len(onet_columns))



# STEP 2: Prepare features and split the data

from sklearn.model_selection import train_test_split

# Gallup background features
numeric_features = ["Q20", "Q51"]

categorical_features = [
    "Q8", "Q9", "Q14", "Q16", "Q17", "Q40"
]

# Combine background and O*NET numeric features
all_numeric_features = numeric_features + onet_columns

# Define predictors and target
X = df[all_numeric_features + categorical_features].copy()
y = df["satisfied"]

# Use the same split as Models 1 and 2
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTUNED MODEL TRAIN/TEST SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Numeric features:", len(all_numeric_features))
print("Categorical features:", len(categorical_features))



# STEP 3: Create preprocessing pipelines

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

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, all_numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("\nTUNED MODEL PREPROCESSING READY")
print("Numeric features:", len(all_numeric_features))
print("Categorical features:", len(categorical_features))



# STEP 4: Tune logistic regression using cross-validation

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Build logistic regression pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# Regularization strengths to test
parameters = {
    "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0]
}

# Use 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Find the best regularization strength
grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

print("\nSTARTING CROSS-VALIDATION")

grid_search.fit(X_train, y_train)

print("\nCROSS-VALIDATION RESULTS")

for c, score in zip(
    grid_search.cv_results_["param_classifier__C"],
    grid_search.cv_results_["mean_test_score"]
):
    print(f"C = {c}: Mean ROC-AUC = {score:.4f}")

print("\nBEST REGULARIZATION STRENGTH")
print("Best C:", grid_search.best_params_["classifier__C"])
print("Best cross-validation ROC-AUC:", round(grid_search.best_score_, 4))



# STEP 5: Evaluate the best model on the test set

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Get the best model from cross-validation
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
auc = roc_auc_score(y_test, y_prob)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nTUNED MODEL TEST RESULTS")
print("Best C:", grid_search.best_params_["classifier__C"])
print("ROC-AUC:", round(auc, 4))
print("Balanced accuracy:", round(balanced_acc, 4))
print("F1 score:", round(f1, 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
