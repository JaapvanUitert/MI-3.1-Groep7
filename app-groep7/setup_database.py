"""Create the SQLite database file longkanker.db from database.sql.

Run:  python setup_database.py

Running it again wipes the table and starts fresh, so you can edit
database.sql and re-run this script as often as you like.
"""
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
SQL_FILE = HERE / "database.sql"
DB_FILE = HERE / "longkanker.db"

with sqlite3.connect(DB_FILE) as conn:
    conn.executescript(SQL_FILE.read_text(encoding="utf-8"))
    rows = conn.execute("SELECT id, display_name FROM patient ORDER BY id").fetchall()

print(f"Database {DB_FILE.name} succesvol aangemaakt met {len(rows)} patiënten:")
for patient_id, name in rows:
    print(f"  Patiënt ID {patient_id}: {name}")
