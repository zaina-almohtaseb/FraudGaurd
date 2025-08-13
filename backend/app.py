# backend/app.py
from __future__ import annotations

import os, re
from datetime import datetime

from flask import Flask, jsonify, request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from backend.db import (
    init_db, insert_raw, insert_clean, update_transaction_label,
    get_last_trained_record, save_model_metadata, get_conn,
    count_new_labeled_since
)
from backend.model_runner import predict_proba, expected_features, load_model
from backend.preprocess import preprocess_input, CATEGORIES as UI_CATEGORIES, GENDERS as UI_GENDERS
from backend.retraining import retrain_single_model

# -----------------------------------------------------------------------------
# Config & app init
# -----------------------------------------------------------------------------
load_dotenv()

ADMIN_TOKEN        = os.getenv("ADMIN_TOKEN", "changeme")
RETRAIN_THRESHOLD  = int(os.getenv("RETRAIN_THRESHOLD", "1000"))
CHECK_INTERVAL_MIN = int(os.getenv("CHECK_INTERVAL_MIN", "5"))
FRONTEND_ORIGIN    = os.getenv("FRONTEND_ORIGIN", "").strip()  # e.g., http://127.0.0.1:5500

app = Flask(__name__)

# CORS: if FRONTEND_ORIGIN is set, restrict to it; else default-open (helps file:// testing)
if FRONTEND_ORIGIN:
    CORS(app, resources={r"/*": {"origins": [FRONTEND_ORIGIN]}})
else:
    CORS(app)

# Ensure DB exists
init_db()

# Try load model on startup (okay if first run and model doesn't exist yet)
try:
    load_model()
except Exception as e:
    print(f"[Startup] Model not loaded yet: {e}")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def auth_ok(req) -> bool:
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    token = auth.split(" ", 1)[1]
    return token == ADMIN_TOKEN

def _latest_model_metadata():
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT model_version, trained_on, last_trained_record, auc, rows_trained, train_seconds "
                "FROM model_metadata ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            return {
                "model_version": row["model_version"],
                "trained_on": row["trained_on"],
                "last_trained_record": int(row["last_trained_record"]),
                "auc": row["auc"],
                "rows_trained": row["rows_trained"],
                "train_seconds": row["train_seconds"],
            }
    except Exception as e:
        print(f"[ModelInfo] ERROR: {e}")
        return None

# -------- server-side validation --------
_INT = re.compile(r"^-?\d+$")
_ZIP = re.compile(r"^\d{3,10}$")
_MERCHANT = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")

_ALLOWED_AGE = {str(i) for i in range(0, 9)} | {"U"}
_ALLOWED_GENDER = set(UI_GENDERS) | {"U"}
_ALLOWED_CAT = set(UI_CATEGORIES)

def validate_payload(p: dict) -> dict:
    """
    Returns {} when valid; else dict of field -> error message.
    """
    errs = {}

    # amount
    amt = p.get("amount", None)
    try:
        amt = float(amt)
        if amt < 0:
            errs["amount"] = "must be ≥ 0"
    except Exception:
        errs["amount"] = "must be a number"

    # step
    step = p.get("step", None)
    try:
        # allow floats like "10.0" but coerce to int >= 0
        s = int(float(step))
        if s < 0:
            errs["step"] = "must be an integer ≥ 0"
        else:
            p["step"] = s
    except Exception:
        errs["step"] = "must be an integer ≥ 0"

    # age
    age = str(p.get("age", "U")).strip()
    if age not in _ALLOWED_AGE:
        errs["age"] = f"must be one of {sorted(_ALLOWED_AGE)}"
    p["age"] = age

    # gender
    gender = str(p.get("gender", "U")).strip().upper()
    if gender not in _ALLOWED_GENDER:
        errs["gender"] = f"must be one of {sorted(_ALLOWED_GENDER)}"
    p["gender"] = gender

    # category
    cat = str(p.get("category", "")).strip().lower()
    if cat not in _ALLOWED_CAT:
        errs["category"] = f"must be one of {sorted(_ALLOWED_CAT)}"
    p["category"] = cat

    # optional strings (basic sanity)
    merch = p.get("merchant")
    if merch:
        if not _MERCHANT.match(str(merch)):
            errs["merchant"] = "alphanumeric/_/- up to 32 chars"

    zc1 = p.get("zipcodeOri")
    if zc1 and not _ZIP.match(str(zc1)):
        errs["zipcodeOri"] = "digits (3–10)"

    zc2 = p.get("zipMerchant")
    if zc2 and not _ZIP.match(str(zc2)):
        errs["zipMerchant"] = "digits (3–10)"

    return errs


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})

@app.get("/model/info")
def model_info():
    meta = _latest_model_metadata() or {}
    last_id = meta.get("last_trained_record", 0)
    try:
        new_labeled = count_new_labeled_since(last_id)
    except Exception:
        new_labeled = None

    return jsonify({
        "expected_features": expected_features(),
        "retrain_threshold": RETRAIN_THRESHOLD,
        "check_interval_min": CHECK_INTERVAL_MIN,
        "model_version": meta.get("model_version"),
        "last_trained": meta.get("trained_on"),
        "last_trained_record": last_id,
        "new_labeled": new_labeled,
        # new bits:
        "auc": meta.get("auc"),
        "rows_trained": meta.get("rows_trained"),
        "train_seconds": meta.get("train_seconds"),
    })


@app.post("/predict")
def api_predict():
    """
    Predict AND persist:
      - validates payload
      - stores raw payload
      - predicts
      - stores clean feature vector (and optional label if provided)
    """
    payload = request.get_json(silent=True) or {}
    errs = validate_payload(payload)
    if errs:
        return jsonify({"status": "error", "errors": errs}), 400

    try:
        # 1) store raw
        raw_id = insert_raw(payload)

        # 2) predict
        res = predict_proba(payload)

        # 3) store clean features
        X = preprocess_input(payload)
        features = {col: float(X.iloc[0][col]) for col in X.columns}
        fraud_label = payload.get("fraud")  # optional
        insert_clean(raw_id, features, fraud_label)

        # 4) labeled-row counters
        last_id = get_last_trained_record()
        new_labeled = count_new_labeled_since(last_id)

        return jsonify({
            "status": "ok",
            "raw_id": raw_id,
            **res,
            "retrain": {
                "threshold": RETRAIN_THRESHOLD,
                "new_labeled": new_labeled,
                "new_records": new_labeled,  # for backward-compat with old UI
                "should_retrain": (new_labeled is not None and new_labeled >= RETRAIN_THRESHOLD)
            }
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.post("/transactions")
def api_transactions():
    return api_predict()

@app.post("/label")
def api_label():
    # Protect with admin token
    if not auth_ok(request):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    rid = data.get("id")
    fraud = data.get("fraud")
    if rid is None or fraud not in (0, 1, "0", "1"):
        return jsonify({"status": "error", "message": "Provide 'id' and binary 'fraud' (0 or 1)."}), 400
    try:
        updated = update_transaction_label(int(rid), int(fraud))
        if not updated:
            return jsonify({"status": "error", "message": f"id {rid} not found"}), 404
        return jsonify({"status": "ok", "id": int(rid), "fraud": int(fraud)}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.post("/retrain")
def api_retrain():
    if not auth_ok(request):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    try:
        model_version, used_id, m = retrain_single_model()
        save_model_metadata(
        model_version,
        last_record_id=used_id,
        threshold=RETRAIN_THRESHOLD,
        auc=m.get("auc"),
        rows_trained=m.get("rows_trained"),
        train_seconds=m.get("train_seconds"),
)

        return jsonify({
            "status": "ok",
            "model_version": model_version,
            "last_trained_record": used_id
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    
@app.get("/metrics")
def metrics():
    """
    Lightweight metrics for demos:
    - total labeled rows
    - class balance (0/1)
    - quick AUC on an 80/20 stratified split (if possible)
    """
    try:
        from backend.db import fetch_all_clean_as_df
        from backend.preprocess import preprocess_training
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        from sklearn.ensemble import GradientBoostingClassifier
        import numpy as np

        df = fetch_all_clean_as_df()
        df = df.dropna(subset=["fraud"]).copy()
        total_labeled = int(len(df))
        n0 = int((df["fraud"] == 0).sum()) if total_labeled else 0
        n1 = int((df["fraud"] == 1).sum()) if total_labeled else 0

        auc = None
        if total_labeled >= 4 and min(n0, n1) >= 2:
            X, y = preprocess_training(df)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            # Small, fast model just for metric; doesn’t touch deployed model
            clf = GradientBoostingClassifier(random_state=42)
            clf.fit(Xtr, ytr)
            proba = clf.predict_proba(Xte)[:, 1]
            auc = float(roc_auc_score(yte, proba))

        return jsonify({
            "total_labeled": total_labeled,
            "class_0": n0,
            "class_1": n1,
            "auc_sample": auc,   # can be null when too few labels
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
import glob, shutil

@app.get("/model/versions")
def model_versions():
    """List saved versioned model files (requires no auth for demo)."""
    files = sorted(
        [os.path.basename(p) for p in glob.glob(os.path.join(os.path.dirname(__file__), "..", "model", "gb-*.pkl"))],
        reverse=True
    )
    return jsonify({"versions": files})

@app.post("/model/use")
def model_use():
    """Switch deployed model to a specific versioned file (auth required)."""
    if not auth_ok(request):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    version = data.get("version", "")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    src_pkl = os.path.join(model_dir, version)
    src_cols = os.path.join(model_dir, f"feature_columns-{version.replace('.pkl','')}.json")
    if not os.path.exists(src_pkl):
        return jsonify({"status": "error", "message": f"Model file not found: {version}"}), 404
    # copy to "current"
    dst_pkl = os.path.join(model_dir, "gb_model.pkl")
    dst_cols = os.path.join(model_dir, "feature_columns.json")
    try:
        shutil.copy2(src_pkl, dst_pkl)
        if os.path.exists(src_cols):
            shutil.copy2(src_cols, dst_cols)
        load_model()  # hot-reload
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "ok", "using": version})


# -----------------------------------------------------------------------------
# Background auto-retraining (uses LABELED rows)
# -----------------------------------------------------------------------------
def _maybe_retrain():
    try:
        last_id = get_last_trained_record()
        new_labeled = count_new_labeled_since(last_id)

        if new_labeled is not None and new_labeled >= RETRAIN_THRESHOLD:
            print(f"[AutoRetrain] labeled threshold met ({new_labeled} >= {RETRAIN_THRESHOLD}). Retraining...")
            model_version, used_id, m = retrain_single_model()
            save_model_metadata(
                model_version,
                last_record_id=used_id,
                threshold=RETRAIN_THRESHOLD,
                auc=m.get("auc"),
                rows_trained=m.get("rows_trained"),
                train_seconds=m.get("train_seconds"),)

            print(f"[AutoRetrain] retrained -> {model_version}, last_id={used_id}")
        else:
            print(f"[AutoRetrain] waiting for labels ({new_labeled or 0}/{RETRAIN_THRESHOLD}).")
    except Exception as e:
        print(f"[AutoRetrain] ERROR: {e}")

def _start_scheduler():
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(
        _maybe_retrain,
        "interval",
        minutes=CHECK_INTERVAL_MIN,
        id="auto_retrain",
        replace_existing=True
    )
    sched.start()
    print(f"[Scheduler] started; interval={CHECK_INTERVAL_MIN} min")
    return sched

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Only start scheduler in the main process (Flask debug spawns a reloader process)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        _start_scheduler()
    app.run(host="127.0.0.1", port=5000, debug=True)
