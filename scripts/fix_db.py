# scripts/fix_db.py
import sqlite3, os
db = os.path.join('data','fraud.db')
conn = sqlite3.connect(db)
conn.execute("ALTER TABLE transactions_raw ADD COLUMN payload_json TEXT;")
conn.commit()
print("Added payload_json to", db)
print("transactions_raw columns:")
for r in conn.execute("PRAGMA table_info(transactions_raw)"):
    print(r)
conn.close()
