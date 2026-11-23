'''
Neural Network model Test attempt 1 
Data: O*NET + GSS 
To prevent overfitting, keeping neural network depth shallow 
and regularized. 

Architecture: A 2-hidden-layer Multi-Layer Perceptron (64 -> 32 -> 1) 
with ReLU activations and a sigmoid output for binary probabilities.

Regularization: Dropout (0.3 and 0.2) plus EarlyStopping monitoring validation AUC 
so training stops the moment it starts memorizing noise.

Apples-to-Apples Comparison: Using the exact same random_state=42 split, 
SimpleImputer, StandardScaler (required for NNs), and class_weight balancing 
as Logistic Regression workflow script.

'''

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, mean_squared_error, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load Data (Make sure this is your >= 3.0 threshold dataset)
data_path = Path(__file__).resolve().parent.parent / 'final_ml_dataset.csv'
df = pd.read_csv(data_path).dropna(subset=['SATJOB_BINARY'])
X = df.drop(columns=['SATJOB_BINARY'])
y = df['SATJOB_BINARY'].astype(int)

# 2. Identical Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Impute & Scale (Required for Neural Networks)
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_test_sc = scaler.transform(X_test_imp)

# 4. Compute Balanced Class Weights (Matches class_weight='balanced' in sklearn)
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: weights[0], 1: weights[1]}

from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 5. Build Tightly Regularized Tabular MLP
model = Sequential([
    Input(shape=(X_train_sc.shape[1],)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=1e-4),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc')]
)

# 6. Callbacks: Decay LR on plateau, then stop early
lr_scheduler = ReduceLROnPlateau(
    monitor='val_auc',
    mode='max',
    factor=0.5,
    patience=4,
    min_lr=1e-5,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_auc',
    mode='max',
    patience=12,
    restore_best_weights=True
)

# 7. Train with smoother batch size
history = model.fit(
    X_train_sc, y_train,
    validation_split=0.2,
    epochs=120,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=[lr_scheduler, early_stop],
    verbose=1
)

# 8. Evaluate on Test Set
nn_pred_prob = model.predict(X_test_sc).ravel()
nn_pred_class = (nn_pred_prob >= 0.5).astype(int)

nn_auc = roc_auc_score(y_test, nn_pred_prob)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred_prob))
nn_f1 = f1_score(y_test, nn_pred_class)

print("\n--- Keras Neural Network Results ---")
print(f"AUC:  {nn_auc:.4f}")
print(f"RMSE: {nn_rmse:.4f}")
print(f"F1:   {nn_f1:.4f}")

np.save('nn_probs.npy', nn_pred_prob)

