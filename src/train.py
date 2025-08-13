# src/train.py
from __future__ import annotations

import os, json, joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from preprocess import preprocess_training, FEATURE_COLUMNS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "fraud - fraud.csv")  # adjust if needed
MODEL_DIR  = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "gb_model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

os.makedirs(MODEL_DIR, exist_ok=True)

def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize common oddities in this dataset (trailing quotes)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.rstrip("'")  # e.g., "M'" -> "M"
    # Ensure required columns exist (fill with safe defaults if missing)
    for need in ["amount", "step", "age", "gender", "category"]:
        if need not in df.columns:
            df[need] = None
    # label column
    if "fraud" not in df.columns:
        # sometimes named isFraud
        if "isFraud" in df.columns:
            df["fraud"] = df["isFraud"]
        else:
            raise RuntimeError("Dataset must contain a 'fraud' (or 'isFraud') column.")
    return df

def main():
    df = _load_csv(DATA_PATH)
    X, y = preprocess_training(df)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    print(f"Test AUC: {auc:.4f}")
    print(classification_report(yte, (proba >= 0.5).astype(int)))

    joblib.dump(clf, MODEL_PATH)
    with open(COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved feature columns to {COLUMNS_PATH}")

if __name__ == "__main__":
    main()
