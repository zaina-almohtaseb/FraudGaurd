# src/preprocess.py
from __future__ import annotations
import pandas as pd

CATEGORIES = [
    "es_transport", "es_food", "es_health",
    "es_fashion", "es_home", "es_entertainment", "es_others"
]
GENDERS = ["M", "F"]

BASE_NUMERIC = ["amount", "step", "age_num"]
GENDER_OH    = [f"gender_{g}" for g in GENDERS]
CAT_OH       = [f"cat_{c}" for c in CATEGORIES]
FEATURE_COLUMNS = BASE_NUMERIC + GENDER_OH + CAT_OH

def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return int(default)

def _clean_str(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.endswith("'"):
        s = s[:-1]
    return s

def _norm_gender(g: str) -> str:
    g = _clean_str(g).upper()
    return g if g in GENDERS else "U"

_CAT_MAP = {
    "es_transportation": "es_transport",
    "es_transport": "es_transport",
    "es_food": "es_food",
    "es_health": "es_health",
    "es_fashion": "es_fashion",
    "es_home": "es_home",
    "es_entertainment": "es_entertainment",
    "es_others": "es_others",
}
def _norm_category(c: str) -> str:
    c = _clean_str(c).lower()
    if c in _CAT_MAP:
        return _CAT_MAP[c]
    if c.startswith("es_transport"): return "es_transport"
    if c.startswith("es_food"): return "es_food"
    if c.startswith("es_health"): return "es_health"
    if c.startswith("es_fashion"): return "es_fashion"
    if c.startswith("es_home"): return "es_home"
    if c.startswith("es_entertainment"): return "es_entertainment"
    return "es_others"

def _age_to_num(a: object) -> int:
    s = _clean_str(a)
    if s.upper() == "U" or s == "":
        return -1
    return _to_int(s, default=-1)

def _vector_from_row(row: dict) -> dict:
    amount = _to_float(row.get("amount", 0.0), 0.0)
    step   = max(0, _to_int(row.get("step", 0), 0))
    age    = _age_to_num(row.get("age", row.get("age_band", row.get("age_num", "U"))))
    gender = _norm_gender(row.get("gender", "U"))
    cat    = _norm_category(row.get("category", ""))

    feats = {
        "amount": amount,
        "step": step,
        "age_num": age,
        **{k: 0 for k in GENDER_OH},
        **{k: 0 for k in CAT_OH},
    }
    if gender in ["M","F"]:
        feats[f"gender_{gender}"] = 1
    feats[f"cat_{cat}"] = 1
    return feats

def preprocess_training(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=int)
    rows = []
    for _, r in df.iterrows():
        rows.append(_vector_from_row(r.to_dict()))
    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    if "fraud" not in df.columns:
        raise RuntimeError("preprocess_training: expected a 'fraud' column.")
    y = df["fraud"].astype("Int64").fillna(0).astype(int)
    return X, y
