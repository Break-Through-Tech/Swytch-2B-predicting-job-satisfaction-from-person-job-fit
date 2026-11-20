import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error

# 1. Load the ML-ready data
df = pd.read_csv('final_ml_dataset.csv')
df = df.dropna(subset=['SATJOB_BINARY'])

# Separate Features (X) and Target (y)
X = df.drop(columns=['SATJOB_BINARY'])
y = df['SATJOB_BINARY']

# 2. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. IMPUTE MISSING VALUES (Notice we skip the StandardScaler step here)
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 4. TRAIN THE RANDOM FOREST
rf_model = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced_subsample', 
    random_state=42,
    n_jobs=-1,
    max_depth=8, 
    min_samples_leaf=15
)
rf_model.fit(X_train_imputed, y_train)

# 5. GENERATE PREDICTIONS
rf_pred_prob = rf_model.predict_proba(X_test_imputed)[:, 1]
rf_pred_class = rf_model.predict(X_test_imputed)

# 6. CALCULATE METRICS
rf_auc = roc_auc_score(y_test, rf_pred_prob)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred_prob))
rf_f1 = f1_score(y_test, rf_pred_class)

print("--- Random Forest Results ---")
print(f"AUC:  {rf_auc:.4f}")
print(f"RMSE: {rf_rmse:.4f}")
print(f"F1:   {rf_f1:.4f}")

# 7. EXTRACT FEATURE IMPORTANCES
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Top 10 Most Important Features (Random Forest) ---")
print(importance_df.head(10).to_string(index=False))