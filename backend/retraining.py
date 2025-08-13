from __future__ import annotations
import os, json, shutil, time
from datetime import datetime

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from backend.db import fetch_all_clean_as_df, get_clean_since, get_last_trained_record
from backend.model_runner import load_model
from backend.preprocess import preprocess_training

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "gb_model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

def _save_versioned(clf, columns, model_version: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    version_pkl  = os.path.join(MODEL_DIR, f"{model_version}.pkl")
    version_cols = os.path.join(MODEL_DIR, f"feature_columns-{model_version}.json")
    joblib.dump(clf, version_pkl)
    with open(version_cols, "w", encoding="utf-8") as f:
        json.dump(list(columns), f)
    # deploy current
    shutil.copy2(version_pkl, MODEL_PATH)
    shutil.copy2(version_cols, COLUMNS_PATH)

def retrain_single_model():
    """
    Windowed retrain:
      - try using only rows labeled since last training
      - if too small, fall back to all labeled rows
      - compute a quick AUC on a holdout IF possible (doesn't affect the deployed model)
      - always train final model on ALL rows in the window
    Returns: (model_version, last_raw_id_used, metrics_dict)
    """
    t0 = time.time()
    last_id = get_last_trained_record()

    df = get_clean_since(last_id)
    if df.empty:
        df = fetch_all_clean_as_df()
    if df.empty:
        raise RuntimeError("No clean labeled data available to retrain.")
    df = df.dropna(subset=["fraud"]).copy()
    if df.empty:
        raise RuntimeError("No labeled rows to train on.")

    X, y = preprocess_training(df)
    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        raise RuntimeError("Need labels for both classes (0 and 1). Label a few more rows.")

    # Optional holdout AUC for the window (only if data supports it)
    auc = None
    try:
        vc0 = (y == 0).sum()
        vc1 = (y == 1).sum()
        if len(y) >= 10 and min(vc0, vc1) >= 2:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            tmp = GradientBoostingClassifier(random_state=42).fit(Xtr, ytr)
            auc = float(roc_auc_score(yte, tmp.predict_proba(Xte)[:, 1]))
    except Exception as e:
        print(f"[Retrain] AUC skipped: {e}")

    # Final model trained on ALL rows in the window
    clf = GradientBoostingClassifier(random_state=42).fit(X, y)

    model_version = f"gb-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    _save_versioned(clf, X.columns, model_version)
    load_model()

    last_raw_id_used = int(df["raw_id"].max()) if "raw_id" in df.columns else 0
    train_seconds = round(time.time() - t0, 3)
    metrics = {"auc": auc, "rows_trained": int(len(y)), "train_seconds": train_seconds}

    print(f"[Retrain] {model_version} | rows={len(y)} classes={classes} auc={auc} time={train_seconds}s (last_id={last_raw_id_used})")
    return model_version, last_raw_id_used, metrics
