import sqlite3

DB_NAME = 'attendance.db'

def create_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        confidence REAL
    )
    """)

    conn.commit()
    conn.close()

def mark_attendance(name, date, time, confidence):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO attendance (name, date, time, confidence)
    VALUES (?, ?, ?, ?)
    """, (name, date, time, confidence))

    conn.commit()
    conn.close()

def attendance_exists(name, date):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM attendance WHERE name = ? AND date = ?
    """, (name, date))

    result = cursor.fetchone()
    conn.close()
    return result is not None
                   

