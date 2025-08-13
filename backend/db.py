# backend/db.py
from __future__ import annotations

import json
import os
import sqlite3
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.getenv("APP_DB_PATH", os.path.join(DATA_DIR, "fraud.db"))
os.makedirs(DATA_DIR, exist_ok=True)

# Structured feature names we support (match preprocess)
GENDERS = ("M", "F", "U")
CATEGORIES = (
    "es_transport", "es_food", "es_health", "es_fashion",
    "es_home", "es_entertainment", "es_others",
)

# ---------------------------------------------------------------------
# Schema for brand-new DBs
# NOTE: Only create indexes here that are safe for *any* existing schema.
# Indexes that reference newly added columns will be created later in
# _ensure_columns() **after** columns are present.
# ---------------------------------------------------------------------
_SQL_INIT = f"""
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS transactions_raw (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    payload_json TEXT NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transactions_clean (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id         INTEGER,
    features_json  TEXT NOT NULL,
    fraud          INTEGER,

    -- structured numeric columns
    amount         REAL,
    amount_num     REAL,
    step           INTEGER,
    age_num        INTEGER,

    -- one-hot gender
    {" ,".join([f"gender_{g} INTEGER" for g in GENDERS])},

    -- one-hot category
    {" ,".join([f"cat_{c} INTEGER" for c in CATEGORIES])},

    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metadata (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version        TEXT NOT NULL,
    trained_on           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_trained_record  INTEGER NOT NULL,
    threshold            INTEGER,
    auc                  REAL,
    rows_trained         INTEGER,
    train_seconds        REAL
);

-- Safe indexes (exist across all versions)
CREATE INDEX IF NOT EXISTS idx_raw_created  ON transactions_raw(created_at);
CREATE INDEX IF NOT EXISTS idx_clean_rawid  ON transactions_clean(raw_id);
CREATE INDEX IF NOT EXISTS idx_clean_fraud  ON transactions_clean(fraud);
CREATE INDEX IF NOT EXISTS idx_meta_trained ON model_metadata(trained_on);
"""

# ---------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ---------------------------------------------------------------------
# Init (self-healing for older DBs)
# ---------------------------------------------------------------------
def _cols(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]

def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Bring old databases up to date (adds columns/indexes if missing)."""

    # ---- transactions_raw ----
    rcols = set(_cols(conn, "transactions_raw"))
    if "payload_json" not in rcols:
        conn.execute("ALTER TABLE transactions_raw ADD COLUMN payload_json TEXT;")

    # ---- transactions_clean ----
    ccols = set(_cols(conn, "transactions_clean"))
    def add_clean(col: str, typ: str):
        nonlocal ccols
        if col not in ccols:
            conn.execute(f"ALTER TABLE transactions_clean ADD COLUMN {col} {typ};")
            ccols.add(col)

    add_clean("features_json", "TEXT")
    add_clean("fraud", "INTEGER")
    add_clean("amount", "REAL")
    add_clean("amount_num", "REAL")
    add_clean("step", "INTEGER")
    add_clean("age_num", "INTEGER")

    for g in GENDERS:
        add_clean(f"gender_{g}", "INTEGER")
    for c in CATEGORIES:
        add_clean(f"cat_{c}", "INTEGER")

    # ---- model_metadata ----
    mcols = set(_cols(conn, "model_metadata"))
    def add_meta(col: str, typ: str):
        nonlocal mcols
        if col not in mcols:
            conn.execute(f"ALTER TABLE model_metadata ADD COLUMN {col} {typ};")
            mcols.add(col)

    add_meta("auc", "REAL")
    add_meta("rows_trained", "INTEGER")
    add_meta("train_seconds", "REAL")

    conn.commit()

    # ---- indexes that depend on new columns (guarded) ----
    def has_index(name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
            (name,)
        ).fetchone()
        return bool(row)

    # create only if the column exists and the index doesn't
    if "step" in ccols and not has_index("idx_clean_step"):
        conn.execute("CREATE INDEX idx_clean_step ON transactions_clean(step);")
    if "age_num" in ccols and not has_index("idx_clean_age"):
        conn.execute("CREATE INDEX idx_clean_age ON transactions_clean(age_num);")

    conn.commit()

def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(_SQL_INIT)
        _ensure_columns(conn)

# ---------------------------------------------------------------------
# Inserts / Updates
# ---------------------------------------------------------------------
def insert_raw(payload: Dict[str, Any]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO transactions_raw (payload_json) VALUES (?)",
            (json.dumps(payload, ensure_ascii=False),),
        )
        conn.commit()
        return int(cur.lastrowid)

def insert_clean(raw_id: Optional[int], features: Dict[str, Any], fraud: Optional[int]) -> int:
    """
    Insert clean features for a raw row.
    Always persists features_json; also fills structured numeric / one-hot columns when present.
    """
    def fnum(key: str, default: float = 0.0) -> float:
        try: return float(features.get(key, default))
        except Exception: return default

    def inum(key: str, default: int = 0) -> int:
        try: return int(features.get(key, default))
        except Exception: return default

    with get_conn() as conn:
        table_cols = {r["name"] for r in conn.execute("PRAGMA table_info(transactions_clean)")}

        cols = ["raw_id", "features_json", "fraud"]
        vals = [raw_id, json.dumps(features, ensure_ascii=False), int(fraud) if fraud is not None else None]

        # numeric basics
        if "amount" in table_cols:
            cols.append("amount");      vals.append(fnum("amount"))
        if "amount_num" in table_cols:
            cols.append("amount_num");  vals.append(fnum("amount"))
        if "step" in table_cols:
            cols.append("step");        vals.append(inum("step"))
        if "age_num" in table_cols:
            cols.append("age_num");     vals.append(inum("age_num"))

        # one-hot genders
        for g in GENDERS:
            col = f"gender_{g}"
            if col in table_cols:
                cols.append(col); vals.append(inum(col))
        # one-hot categories
        for c in CATEGORIES:
            col = f"cat_{c}"
            if col in table_cols:
                cols.append(col); vals.append(inum(col))

        placeholders = ",".join("?" for _ in cols)
        sql = f"INSERT INTO transactions_clean ({','.join(cols)}) VALUES ({placeholders})"
        cur = conn.execute(sql, tuple(vals))
        conn.commit()
        return int(cur.lastrowid)

def update_transaction_label(raw_id: int, fraud: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE transactions_clean SET fraud=? WHERE raw_id=?",
            (int(fraud), int(raw_id)),
        )
        conn.commit()
        return cur.rowcount > 0

# ---------------------------------------------------------------------
# Metadata / Counters
# ---------------------------------------------------------------------
def get_last_trained_record() -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT last_trained_record FROM model_metadata ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return int(row["last_trained_record"]) if row else 0

def count_new_records(since_id: int) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM transactions_raw WHERE id > ?",
            (int(since_id),),
        ).fetchone()
        return int(row["c"]) if row else 0

def count_new_labeled_since(last_raw_id: int) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM transactions_clean WHERE raw_id > ? AND fraud IS NOT NULL",
            (int(last_raw_id),),
        ).fetchone()
        return int(row["c"]) if row else 0

def save_model_metadata(
    model_version: str,
    last_record_id: int,
    threshold: int,
    auc: float | None = None,
    rows_trained: int | None = None,
    train_seconds: float | None = None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO model_metadata
              (model_version, last_trained_record, threshold, auc, rows_trained, train_seconds)
            VALUES (?,?,?,?,?,?)
            """,
            (model_version, int(last_record_id), int(threshold), auc, rows_trained, train_seconds),
        )
        conn.commit()
        return int(cur.lastrowid)

# ---------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------
def _row_to_feature_dict(r: sqlite3.Row) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    if r["features_json"]:
        try: feats = json.loads(r["features_json"])
        except Exception: feats = {}

    # numerics
    if "amount_num" in r.keys() and r["amount_num"] is not None:
        feats["amount"] = r["amount_num"]
    elif "amount" in r.keys() and r["amount"] is not None:
        feats["amount"] = r["amount"]
    if "step" in r.keys() and r["step"] is not None:
        feats["step"] = r["step"]
    if "age_num" in r.keys() and r["age_num"] is not None:
        feats["age_num"] = r["age_num"]

    # one-hots
    for g in GENDERS:
        k = f"gender_{g}"
        if k in r.keys() and r[k] is not None:
            feats[k] = r[k]
    for c in CATEGORIES:
        k = f"cat_{c}"
        if k in r.keys() and r[k] is not None:
            feats[k] = r[k]
    return feats

def fetch_all_clean_as_df():
    import pandas as pd
    rows: List[Dict[str, Any]] = []
    with get_conn() as conn:
        for r in conn.execute(
            f"""
            SELECT id, raw_id, features_json, fraud, amount, amount_num, step, age_num,
                   {",".join([f"gender_{g}" for g in GENDERS])},
                   {",".join([f"cat_{c}" for c in CATEGORIES])},
                   created_at
            FROM transactions_clean
            ORDER BY id ASC
            """
        ):
            feats = _row_to_feature_dict(r)
            rows.append({"raw_id": r["raw_id"], "fraud": r["fraud"], "created_at": r["created_at"], **feats})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def get_clean_since(last_raw_id: int):
    import pandas as pd
    rows: List[Dict[str, Any]] = []
    with get_conn() as conn:
        for r in conn.execute(
            f"""
            SELECT id, raw_id, features_json, fraud, amount, amount_num, step, age_num,
                   {",".join([f"gender_{g}" for g in GENDERS])},
                   {",".join([f"cat_{c}" for c in CATEGORIES])},
                   created_at
            FROM transactions_clean
            WHERE raw_id > ?
            ORDER BY id ASC
            """,
            (int(last_raw_id),),
        ):
            feats = _row_to_feature_dict(r)
            rows.append({"raw_id": r["raw_id"], "fraud": r["fraud"], "created_at": r["created_at"], **feats})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def last_raw_id() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT IFNULL(MAX(id),0) AS mx FROM transactions_raw").fetchone()
        return int(row["mx"]) if row else 0

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DB utilities")
    parser.add_argument("--init", action="store_true", help="Initialize/upgrade the database schema")
    parser.add_argument("--peek", action="store_true", help="Print last few rows from each table")
    args = parser.parse_args()

    if args.init:
        init_db()
        print(f"Initialized/updated DB at {DB_PATH}")

    if args.peek:
        with get_conn() as conn:
            print("\n== transactions_raw (last 5) ==")
            for r in conn.execute("SELECT id, created_at, SUBSTR(payload_json,1,80) AS payload FROM transactions_raw ORDER BY id DESC LIMIT 5"):
                print(dict(r))

            print("\n== transactions_clean (last 5) ==")
            cols = "id, raw_id, fraud, amount, amount_num, step, age_num, created_at"
            for r in conn.execute(f"SELECT {cols} FROM transactions_clean ORDER BY id DESC LIMIT 5"):
                print(dict(r))

            print("\n== model_metadata (last 5) ==")
            for r in conn.execute("SELECT id, model_version, trained_on, last_trained_record, threshold, auc, rows_trained, train_seconds FROM model_metadata ORDER BY id DESC LIMIT 5"):
                print(dict(r))
