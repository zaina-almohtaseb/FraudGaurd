# -*- coding: utf-8 -*-

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score
)
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import joblib


# ------------------------------
# 1) Load data
# ------------------------------
CSV_PATH = "data/fraud - fraud.csv"   # keep the space & dash as in your file name
df = pd.read_csv(CSV_PATH)

print("\nNull counts:\n", df.isnull().sum())

# (Optional) quick class balance plot
plt.figure()
sns.countplot(x='fraud', data=df)
plt.title('Class Distribution: Fraud vs. Not Fraud')
plt.show()


# ------------------------------
# 2) Preprocess (same as your notebook)
# ------------------------------
# Fill missing categorical values like before
df['age'] = df['age'].fillna('U')
df['gender'] = df['gender'].fillna('U')

# Drop rows missing key numeric fields (as you did)
df = df.dropna(subset=['amount', 'step'])

# One-hot encode these categorical columns
df_encoded = pd.get_dummies(df, columns=['age', 'gender', 'category'])

print("\nShape before encoding:", df.shape)
print("Shape after encoding:", df_encoded.shape)

# Keep target
y = df_encoded['fraud']

# Drop columns you excluded during training
drop_cols = ['fraud', 'customer', 'merchant', 'zipcodeOri', 'zipMerchant']
X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns], errors='ignore')

# Keep only numerics (this mirrors your training step)
X = X.select_dtypes(include=[np.number])

# Save the feature columns NOW (while X is still a DataFrame with named columns)
os.makedirs("model", exist_ok=True)
feature_columns = X.columns.tolist()
with open("model/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)
print(f"\nSaved {len(feature_columns)} training columns -> model/feature_columns.json")


# ------------------------------
# 3) Train / Test split + SMOTE (train only)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# ------------------------------
# 4) Train ORIGINAL Gradient Boosting (no tuning)
# ------------------------------
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_res, y_train_res)


# ------------------------------
# 5) Evaluate
# ------------------------------
y_pred = gb_model.predict(X_test)
y_prob = gb_model.predict_proba(X_test)[:, 1]

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)
roc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)

print("\n=== Evaluation (Original Gradient Boosting) ===")
print("Accuracy     :", round(acc, 4))
print("Recall (Fraud):", round(recall, 4))
print("F1 (Fraud)   :", round(f1, 4))
print("ROC-AUC      :", round(roc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Optional: confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Gradient Boosting (Original)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# ------------------------------
# 6) Save model
# ------------------------------
joblib.dump(gb_model, "model/gb_model.pkl")
print("\nOriginal Gradient Boosting model saved to model/gb_model.pkl")
