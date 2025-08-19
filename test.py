import neondb as db
import pandas as pd


df = pd.read_csv("labnetwork_data.csv")

columns_to_keep = [
    "sender",
    "email",
    "subject",
    "body",
    "message_id",
    "thread_id",
    "date_time",
    "cleaned_body",
    "embed",
]

# Keep only the desired columns
df = df[[c for c in columns_to_keep if c in df.columns]]

# Ensure proper types
if "date_time" in df.columns:
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

db.delete_labnetwork_table()

db.create_labnetwork_table()

db.insert_data_to_db(df)

conn = db.connect_db()
cur = conn.cursor()

cur.execute("SELECT * FROM labnetwork")
print(cur.fetchall())

conn.close()
