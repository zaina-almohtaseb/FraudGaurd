# backend/model_runner.py
from __future__ import annotations

import json
import os
import joblib

from backend.preprocess import preprocess_input

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "gb_model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

model = None
TRAIN_COLUMNS = None

def load_model():
    """Load model and expected training columns."""
    global model, TRAIN_COLUMNS
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")
    model = joblib.load(MODEL_PATH)
    if os.path.exists(COLUMNS_PATH):
        with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
            TRAIN_COLUMNS = json.load(f)
    else:
        TRAIN_COLUMNS = None
    return model

def ensure_loaded():
    global model
    if model is None:
        load_model()

def predict_proba(payload: dict) -> dict:
    """
    Returns:
      {"fraud_prediction": 0|1, "fraud_probability": float in [0,1]}
    """
    ensure_loaded()
    X = preprocess_input(payload)
    proba = float(model.predict_proba(X)[0][1])
    pred = int(proba >= 0.5)  # tune threshold as needed
    return {"fraud_prediction": pred, "fraud_probability": proba}

def expected_features():
    return list(TRAIN_COLUMNS or [])
