import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_percentage_error

# 1. Load the ML-ready data
df = pd.read_csv('final_ml_dataset.csv')

# Drop any rows where the target itself is missing (we can't train on a blank answer)
df = df.dropna(subset=['SATJOB_BINARY'])

# 2. Separate Features (X) and Target (y)
X = df.drop(columns=['SATJOB_BINARY'])
y = df['SATJOB_BINARY']

# 3. TRAIN/TEST SPLIT FIRST 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. IMPUTE MISSING VALUES using median 
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 5. SCALE THE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 6. TRAIN THE BASELINE MODEL
# class_weight='balanced' for survey imbalance
# max_iter=1000 ensures the underlying solver converges without throwing warnings
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# GENERATE PREDICTIONS
# probabilities (0.0 to 1.0) as error metrics
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

auc = roc_auc_score(y_test, y_pred_prob)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_prob))
mape = mean_absolute_percentage_error(y_test, y_pred_prob)

print(f"--- Baseline Logistic Regression Results ---")
print(f"AUC:  {auc:.4f} (1.0 is perfect, 0.5 is random guessing)")
print(f"RMSE: {rmse:.4f} (Lower is better)")
print(f"MAPE: {mape:.4f} (Lower is better)")