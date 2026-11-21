
# STEP 1: Load the Gallup + O*NET dataset

from pathlib import Path
import pandas as pd

# Find the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Find the dataset
data_path = (
    project_folder
    / "notebooks"
    / "gallup_eda"
    / "output"
    / "gallup_fsu_pre_model.csv"
)

# Load the data
df = pd.read_csv(data_path)

# Identify all O*NET features
onet_columns = [
    col for col in df.columns
    if col.startswith("onet_")
]

# Define Gallup background features
numeric_features = ["Q20", "Q51"]

categorical_features = [
    "Q8", "Q9", "Q14", "Q16", "Q17", "Q40"
]

# Combine numeric background and O*NET features
all_numeric_features = numeric_features + onet_columns

print("\nRANDOM FOREST DATASET")
print("Rows:", len(df))
print("O*NET features:", len(onet_columns))
print("Total features:", len(all_numeric_features) + len(categorical_features))



# STEP 2: Prepare features and split the data

from sklearn.model_selection import train_test_split

# Define predictors and target
X = df[all_numeric_features + categorical_features].copy()
y = df["satisfied"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nRANDOM FOREST TRAIN/TEST SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Numeric features:", len(all_numeric_features))
print("Categorical features:", len(categorical_features))



# STEP 3: Preprocess features for Random Forest

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Fill missing numeric values with the median
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Fill missing categorical values and encode categories
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine numeric and categorical preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, all_numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("\nRANDOM FOREST PREPROCESSING READY")
print("Numeric features:", len(all_numeric_features))
print("Categorical features:", len(categorical_features))


# STEP 4: Tune Random Forest using cross-validation

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# Create Random Forest pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    ))
])

# Hyperparameter values to test
parameters = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_leaf": [1, 5, 10]
}

# Create stratified cross-validation folds
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# Search 8 random parameter combinations
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameters,
    n_iter=8,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=2
)

print("\nSTARTING RANDOM FOREST CROSS-VALIDATION")

random_search.fit(X_train, y_train)

print("\nRANDOM FOREST CROSS-VALIDATION RESULTS")

for score, params in zip(
    random_search.cv_results_["mean_test_score"],
    random_search.cv_results_["params"]
):
    print(f"ROC-AUC: {score:.4f} | {params}")

print("\nBEST RANDOM FOREST SETTINGS")
print("Best parameters:", random_search.best_params_)
print("Best cross-validation ROC-AUC:", round(random_search.best_score_, 4))



# STEP 5: Evaluate the best Random Forest model

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Get the best model from cross-validation
best_model = random_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
auc = roc_auc_score(y_test, y_prob)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nRANDOM FOREST TEST RESULTS")
print("ROC-AUC:", round(auc, 4))
print("Balanced accuracy:", round(balanced_acc, 4))
print("F1 score:", round(f1, 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
