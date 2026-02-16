import sqlite3

DB_NAME = "database/actions.db"

def get_db():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def save_action(action):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO actions (action) VALUES (?)",
        (action,)
    )

    conn.commit()
    conn.close()

