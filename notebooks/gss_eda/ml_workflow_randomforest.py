import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

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

os.makedirs('presentation_visuals', exist_ok=True)

y_pred_prob = np.load('lr_probs.npy')

# Calculate ROC curves and AUCs
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pred_prob)

lr_auc = roc_auc_score(y_test, y_pred_prob)
rf_auc = roc_auc_score(y_test, rf_pred_prob)

# Plot with matching presentation palette
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_auc:.4f})', color='#1D3557', linewidth=2.5)
ax.plot(fpr_rf, tpr_rf, label=f'Tuned Random Forest (AUC = {rf_auc:.4f})', color='#E76F51', linewidth=2.5)
ax.plot([0, 1], [0, 1], linestyle='--', color='#6C757D', label='Random Guessing (AUC = 0.5000)', alpha=0.7)

ax.set_title('ROC Curve Comparison (O*NET >= 3.0 Threshold)', fontsize=13, fontweight='bold')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.legend(loc='lower right', frameon=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('presentation_visuals/05_roc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()