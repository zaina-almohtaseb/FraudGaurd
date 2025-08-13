PRAGMA journal_mode = WAL;

-- Raw submissions from the UI/API
CREATE TABLE IF NOT EXISTS transactions_raw (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    payload_json TEXT NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Clean features + optional label (for retraining)
CREATE TABLE IF NOT EXISTS transactions_clean (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id         INTEGER,                      -- FK to transactions_raw.id (optional)
    features_json  TEXT NOT NULL,                -- numeric vector after preprocessing
    fraud          INTEGER,                      -- optional label (0/1)
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model metadata (for retrain bookkeeping)
CREATE TABLE IF NOT EXISTS model_metadata (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version        TEXT NOT NULL,
    trained_on           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_trained_record  INTEGER NOT NULL,       -- last transactions_raw.id used
    threshold            INTEGER                 -- retrain threshold used
);
