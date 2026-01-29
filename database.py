import sqlite3
import pandas as pd
from datetime import datetime

# This function must be named EXACTLY 'init_db' to match app.py
def init_db():
    """Initializes the local SQLite database for the exhibition."""
    conn = sqlite3.connect('swaas_check.db')
    c = conn.cursor()
    # Create table to store person details and AI results
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  age INTEGER,
                  contact TEXT,
                  prediction TEXT,
                  confidence REAL,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

def save_patient(name, age, contact, prediction, confidence):
    """Saves the specific TB or Normal result directly to the admin log."""
    conn = sqlite3.connect('swaas_check.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO patients (name, age, contact, prediction, confidence, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
              (name, age, contact, prediction, confidence, timestamp))
    conn.commit()
    conn.close()

def get_all_records():
    """Retrieves all stored cough data for the Admin Dashboard."""
    conn = sqlite3.connect('swaas_check.db')
    df = pd.read_sql_query("SELECT * FROM patients ORDER BY timestamp DESC", conn)
    conn.close()
    return df