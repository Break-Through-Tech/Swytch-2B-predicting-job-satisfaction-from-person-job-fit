
# Gallup + O*NET Feature Importance Analysis

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


# STEP 1: Load the dataset

project_folder = Path(__file__).resolve().parent.parent

data_path = (
    project_folder
    / "notebooks"
    / "gallup_eda"
    / "output"
    / "gallup_fsu_pre_model.csv"
)

df = pd.read_csv(data_path)

onet_columns = [
    col for col in df.columns
    if col.startswith("onet_")
]

numeric_features = ["Q20", "Q51"]

categorical_features = [
    "Q8", "Q9", "Q14", "Q16", "Q17", "Q40"
]

all_numeric_features = numeric_features + onet_columns
all_features = all_numeric_features + categorical_features

X = df[all_features].copy()
y = df["satisfied"]

print("\nDATASET SUMMARY")
print("Rows:", len(df))
print("Gallup background features:", 8)
print("O*NET features:", len(onet_columns))
print("Total features:", len(all_features))


# STEP 2: Split the data

# First separate the original test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create validation data from the training set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full
)

print("\nDATA SPLIT")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Reserved test rows:", len(X_test))


# STEP 3: Preprocess features

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, all_numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# STEP 4: Train logistic regression

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nTRAINING LOGISTIC REGRESSION")

model.fit(X_train, y_train)

val_prob = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_prob)

print("Validation ROC-AUC:", round(val_auc, 4))


# STEP 5: Calculate permutation importance

print("\nCALCULATING PERMUTATION IMPORTANCE")
print("This may take several minutes.")

importance = permutation_importance(
    model,
    X_val,
    y_val,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=2
)

results = pd.DataFrame({
    "feature": all_features,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std
})

results["source"] = np.where(
    results["feature"].str.startswith("onet_"),
    "O*NET",
    "Gallup"
)

results = results.sort_values(
    by="importance_mean",
    ascending=False
).reset_index(drop=True)


# STEP 6: Print results

print("\nTOP 20 MOST IMPORTANT FEATURES")

print(
    results.head(20).to_string(index=False)
)

print("\nTOP 10 O*NET FEATURES")

print(
    results[results["source"] == "O*NET"]
    .head(10)
    .to_string(index=False)
)

print("\nTOP 8 GALLUP FEATURES")

print(
    results[results["source"] == "Gallup"]
    .to_string(index=False)
)


# STEP 7: Save results

output_folder = project_folder / "modeling" / "output"
output_folder.mkdir(parents=True, exist_ok=True)

csv_path = output_folder / "gallup_feature_importance.csv"

results.to_csv(csv_path, index=False)

print("\nFEATURE IMPORTANCE RESULTS SAVED")
print("File:", csv_path)


# STEP 8: Create feature importance graph

top_features = results.head(20).sort_values(
    "importance_mean"
)

plt.figure(figsize=(11, 8))

plt.barh(
    top_features["feature"],
    top_features["importance_mean"]
)

plt.xlabel("Mean decrease in validation ROC-AUC")
plt.ylabel("Feature")
plt.title("Top 20 Features: Gallup + O*NET Logistic Regression")

plt.tight_layout()

graph_path = output_folder / "gallup_feature_importance.png"

plt.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nFEATURE IMPORTANCE GRAPH SAVED")
print("File:", graph_path)

print("\nANALYSIS COMPLETE")


# STEP 9: Grouped permutation importance

# Compare the importance of Gallup and O*NET features together.
# We shuffle entire feature groups instead of one column at a time.

print("\nSTARTING GROUPED PERMUTATION IMPORTANCE")

rng = np.random.default_rng(42)

# Use the original validation ROC-AUC
original_auc = roc_auc_score(
    y_val,
    model.predict_proba(X_val)[:, 1]
)

print("Original validation ROC-AUC:", round(original_auc, 4))


# STEP 10: Function to calculate grouped importance

def grouped_importance(model, X_val, y_val, columns, repeats=30):
    auc_drops = []

    for i in range(repeats):

        # Create a copy so the original data stays unchanged
        X_shuffled = X_val.copy()

        # Randomly rearrange respondent rows
        shuffled_indices = rng.permutation(len(X_val))

        # Shuffle all columns in the group together
        X_shuffled.loc[:, columns] = (
            X_val.iloc[shuffled_indices][columns].to_numpy()
        )

        # Predict using the shuffled data
        shuffled_prob = model.predict_proba(X_shuffled)[:, 1]

        # Calculate the new ROC-AUC
        shuffled_auc = roc_auc_score(y_val, shuffled_prob)

        # Measure how much performance decreased
        auc_drop = original_auc - shuffled_auc

        auc_drops.append(auc_drop)

    return {
        "mean_importance": np.mean(auc_drops),
        "std_importance": np.std(auc_drops),
        "min_importance": np.min(auc_drops),
        "max_importance": np.max(auc_drops)
    }


# STEP 11: Calculate importance for each feature group

gallup_importance = grouped_importance(
    model,
    X_val,
    y_val,
    numeric_features + categorical_features
)

onet_importance = grouped_importance(
    model,
    X_val,
    y_val,
    onet_columns
)


# STEP 12: Display results

print("\nGROUPED PERMUTATION IMPORTANCE RESULTS")

print("\nGallup Background Features (8):")
print("Mean AUC decrease:",
      round(gallup_importance["mean_importance"], 4))
print("Standard deviation:",
      round(gallup_importance["std_importance"], 4))
print("Minimum AUC decrease:",
      round(gallup_importance["min_importance"], 4))
print("Maximum AUC decrease:",
      round(gallup_importance["max_importance"], 4))

print("\nO*NET Occupational Features (407):")
print("Mean AUC decrease:",
      round(onet_importance["mean_importance"], 4))
print("Standard deviation:",
      round(onet_importance["std_importance"], 4))
print("Minimum AUC decrease:",
      round(onet_importance["min_importance"], 4))
print("Maximum AUC decrease:",
      round(onet_importance["max_importance"], 4))


# STEP 13: Create comparison table

group_results = pd.DataFrame([
    {
        "feature_group": "Gallup Background",
        "number_of_features": 8,
        **gallup_importance
    },
    {
        "feature_group": "O*NET Occupational",
        "number_of_features": len(onet_columns),
        **onet_importance
    }
])

print("\nFEATURE GROUP COMPARISON")
print(group_results.to_string(index=False))


# STEP 14: Save comparison results

group_csv_path = (
    output_folder / "gallup_grouped_feature_importance.csv"
)

group_results.to_csv(group_csv_path, index=False)

print("\nGROUPED IMPORTANCE CSV SAVED")
print("File:", group_csv_path)


# STEP 15: Create comparison graph

plt.figure(figsize=(9, 6))

plt.bar(
    group_results["feature_group"],
    group_results["mean_importance"],
    yerr=group_results["std_importance"],
    capsize=6
)

plt.ylabel("Mean decrease in validation ROC-AUC")
plt.xlabel("Feature Group")
plt.title("Grouped Permutation Importance: Gallup vs O*NET")

plt.axhline(y=0, color="black", linewidth=0.8)
plt.tight_layout()

group_graph_path = (
    output_folder / "gallup_grouped_feature_importance.png"
)

plt.savefig(
    group_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGROUPED IMPORTANCE GRAPH SAVED")
print("File:", group_graph_path)


# STEP 16: Interpret the results

print("\nGROUPED IMPORTANCE INTERPRETATION")

gallup_score = gallup_importance["mean_importance"]
onet_score = onet_importance["mean_importance"]

if gallup_score > onet_score:
    print("Gallup background features showed greater grouped importance.")
elif onet_score > gallup_score:
    print("O*NET occupational features showed greater grouped importance.")
else:
    print("Both feature groups showed equal mean importance.")

if onet_score > 0:
    print("Shuffling O*NET features reduced validation ROC-AUC on average.")
elif onet_score < 0:
    print("Shuffling O*NET features improved validation ROC-AUC on average.")
else:
    print("Shuffling O*NET features produced no average ROC-AUC change.")

print("\nGROUPED FEATURE IMPORTANCE ANALYSIS COMPLETE")
