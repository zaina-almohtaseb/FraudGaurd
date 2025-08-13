import sqlite3

conn = sqlite3.connect(r"data/fraud.db")
cur = conn.cursor()
print("model_metadata rows:")
for r in cur.execute("""
    select id, model_version, trained_on, last_trained_record, threshold
    from model_metadata
    order by rowid desc
    limit 5
"""):
    print(r)
conn.close()
