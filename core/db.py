import os
import re
import json
import random
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from flask import g, has_app_context
from werkzeug.security import generate_password_hash

DEMO_LECTURERS = [
    "Dr. Okafor",
    "Dr. Adeyemi",
    "Prof. Martins",
    "Dr. Faith",
    "Dr. Pomele",
    "Prof. Balogun",
    "Dr. Chukwu",
    "Dr. (Mrs) Adeleke",
]

DEMO_USERS = [
    ("admin", "admin123", "administrator", None, "System Administrator"),
    ("okafor", "lecturer123", "lecturer", "Dr. Okafor", "Dr. Okafor"),
    ("adeyemi", "lecturer123", "lecturer", "Dr. Adeyemi", "Dr. Adeyemi"),
    ("martins", "lecturer123", "lecturer", "Prof. Martins", "Prof. Martins"),
    ("faith", "lecturer123", "lecturer", "Dr. Faith", "Dr. Faith"),
    ("pomele", "lecturer123", "lecturer", "Dr. Pomele", "Dr. Pomele"),
    ("balogun", "lecturer123", "lecturer", "Prof. Balogun", "Prof. Balogun"),
    ("chukwu", "lecturer123", "lecturer", "Dr. Chukwu", "Dr. Chukwu"),
    ("adeleke", "lecturer123", "lecturer", "Dr. (Mrs) Adeleke", "Dr. (Mrs) Adeleke"),
]


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_dir():
    """Resolve data/ vs Data/ so Linux hosts find the Windows-created folder."""
    root = project_root()
    candidates = [os.path.join(root, name) for name in ("data", "Data")]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "feedback.db")):
            return path
    for path in candidates:
        if os.path.isdir(path):
            return path
    path = os.path.join(root, "data")
    os.makedirs(path, exist_ok=True)
    return path


def data_file(*parts):
    return os.path.join(data_dir(), *parts)


DATABASE_PATH = data_file("feedback.db")

DEFAULT_SETTINGS = {
    "announcement_banner": "2025/2026 Academic Session — Anonymous Student Evaluation of Teaching (SET) Portal Active",
    "show_announcement": "true",
    "landing_title": "Anonymous Lecturer Evaluation",
    "landing_subtitle": "Share numerical ratings and free-text comments. Your name and student identity are never stored.",
    "privacy_notice": "This form does not collect student name, matric number, or login details. Only the lecturer, course, rating, and comment are saved for analysis.",
    "custom_guidelines": "Please evaluate objectively based on course engagement, syllabus delivery, and instructional clarity."
}

def get_connection(db_path=DATABASE_PATH):
    """
    Creates an optimized SQLite connection with:
    - WAL journal mode for concurrent read/write throughput
    - In-memory temporary storage
    - 64MB cache size
    - Row factory for dictionary-like column access
    """
    con = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode = WAL;")
    con.execute("PRAGMA synchronous = NORMAL;")
    con.execute("PRAGMA cache_size = -64000;")
    con.execute("PRAGMA temp_store = MEMORY;")
    con.execute("PRAGMA mmap_size = 268435456;")
    return con

def get_db():
    """Returns thread-local cached database connection."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = get_connection(DATABASE_PATH)
    return db

def init_db(db_path=DATABASE_PATH):
    """Initializes tables, auto-runs schema migrations, and builds B-Tree indexes."""
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    db = get_connection(db_path)
    
    # 1. Base Tables
    db.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            matric_number TEXT,
            lecturer_name TEXT NOT NULL,
            course TEXT NOT NULL,
            course_code TEXT,
            rating INTEGER NOT NULL,
            comment TEXT NOT NULL,
            document_sentiment TEXT,
            aspects_json TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            lecturer_name TEXT,
            full_name TEXT
        )
    ''')
    
    # 2. Schema Migrations (backward-compatibility safe)
    for col in ['student_name TEXT', 'matric_number TEXT', 'document_sentiment TEXT', 'aspects_json TEXT']:
        try:
            db.execute(f'ALTER TABLE feedback ADD COLUMN {col}')
        except Exception:
            pass

    db.execute('''
        CREATE TABLE IF NOT EXISTS site_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    ''')

    # Seed default landing page settings if missing
    for key, val in DEFAULT_SETTINGS.items():
        db.execute('INSERT OR IGNORE INTO site_settings (key, value) VALUES (?, ?)', (key, val))

    # 3. High-Performance B-Tree Indexing
    indexes = [
        ("idx_feedback_lecturer", "feedback(lecturer_name)"),
        ("idx_feedback_course", "feedback(course)"),
        ("idx_feedback_submitted_at", "feedback(submitted_at DESC)"),
        ("idx_feedback_rating", "feedback(rating)"),
        ("idx_users_username", "users(username)"),
        ("idx_users_lecturer_name", "users(lecturer_name)")
    ]
    for idx_name, idx_def in indexes:
        try:
            db.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_def};")
        except Exception:
            pass

    # 4. Default Seed Users
    existing = db.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    if existing == 0:
        for username, password, role, lecturer_name, full_name in DEMO_USERS:
            db.execute(
                'INSERT INTO users (username, password_hash, role, lecturer_name, full_name) VALUES (?, ?, ?, ?, ?)',
                (username, generate_password_hash(password), role, lecturer_name, full_name)
            )
    db.commit()
    db.close()

def get_lecturer_names():
    """Returns list of active lecturer names dynamically from the database."""
    names = []
    try:
        db = get_db()
        rows = db.execute(
            "SELECT DISTINCT lecturer_name FROM users WHERE role = 'lecturer' AND lecturer_name IS NOT NULL"
        ).fetchall()
        for row in rows:
            if row['lecturer_name'] and row['lecturer_name'] not in names:
                names.append(row['lecturer_name'])
        fb_rows = db.execute('SELECT DISTINCT lecturer_name FROM feedback WHERE lecturer_name IS NOT NULL ORDER BY lecturer_name').fetchall()
        for row in fb_rows:
            if row['lecturer_name'] and row['lecturer_name'] not in names:
                names.append(row['lecturer_name'])
    except Exception:
        pass
    return sorted(names) if names else list(DEMO_LECTURERS)

def seed_db_if_empty():
    """Seeds realistic evaluation records if the database is currently empty."""
    db = get_db()
    count = db.execute("SELECT COUNT(*) as c FROM feedback").fetchone()['c']
    synthetic_csv = data_file("synthetic_evaluations.csv")
    if count == 0 and os.path.exists(synthetic_csv):
        try:
            df = pd.read_csv(synthetic_csv)
            for _, row in df.head(150).iterrows():
                db.execute('''
                    INSERT INTO feedback (student_name, matric_number, lecturer_name, course, course_code, rating, comment, document_sentiment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    "Anonymous", "ANON",
                    row['lecturer_name'], row['course'], row.get('course_code', 'CSC'),
                    int(row['rating']), row['comment'], row.get('document_sentiment', 'neutral')
                ))
            db.commit()
        except Exception as e:
            print(f"[WARN] Failed to seed DB: {e}")

def _lecturer_username(full_name, taken):
    tokens = re.findall(r"[A-Za-z]+", full_name or "")
    if not tokens:
        base = "lecturer"
    else:
        base = tokens[-1].lower()
    if base not in taken:
        return base
    initials = "".join(t[0].lower() for t in tokens[:-1])
    candidate = f"{initials}{base}"
    if candidate not in taken:
        return candidate
    n = 2
    while f"{candidate}{n}" in taken:
        n += 1
    return f"{candidate}{n}"

def import_nigerian_live_feedback(analyze_fn=None):
    """
    Loads the 10,000 Nigerian SET comments into the live feedback table once,
    with anonymous student identity, stored analysis, and lecturer login accounts.
    """
    csv_path = data_file("nigerian_lecturer_evaluations_10000.csv")
    if not os.path.exists(csv_path):
        return 0

    owns_connection = False
    if has_app_context():
        db = get_db()
    else:
        db = get_connection()
        owns_connection = True

    try:
        flag = db.execute(
            "SELECT value FROM site_settings WHERE key = ?",
            ("nigerian_live_imported",)
        ).fetchone()
        if flag and str(flag["value"]).lower() == "true":
            return 0
        existing = db.execute("SELECT COUNT(*) AS c FROM feedback").fetchone()["c"]
        if existing >= 9000:
            db.execute(
                "INSERT INTO site_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                ("nigerian_live_imported", "true")
            )
            db.commit()
            return 0

        df = pd.read_csv(csv_path)
        now = datetime.now()
        rows_to_insert = []
        lecturer_names = set()

        for i, row in df.iterrows():
            comment = str(row.get("Comment") or "").strip()
            if not comment or comment.lower() == "nan":
                continue
            lecturer = str(row.get("Lecturer") or "").strip()
            course = str(row.get("Course") or "").strip() or "Computer Science"
            code = str(row.get("Course Code") or "").strip()
            try:
                rating = int(row.get("Rating", 3))
            except (TypeError, ValueError):
                rating = 3
            if lecturer:
                lecturer_names.add(lecturer)

            if analyze_fn:
                ana = analyze_fn(comment, rating)
                sentiment = ana.get("document_sentiment", "neutral")
                aspects_json = json.dumps(ana)
            else:
                sentiment = "negative" if rating <= 2 else ("positive" if rating >= 4 else "neutral")
                aspects_json = None

            submitted_at = (now - timedelta(days=random.randint(0, 150), seconds=random.randint(0, 86400))).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            rows_to_insert.append((
                "Anonymous", "ANON", lecturer, course, code, rating, comment,
                sentiment, aspects_json, submitted_at
            ))
            if (i + 1) % 1000 == 0:
                print(f"[INFO] Prepared {i + 1} Nigerian live evaluations...", flush=True)

        db.executemany(
            '''INSERT INTO feedback (
                student_name, matric_number, lecturer_name, course, course_code,
                rating, comment, document_sentiment, aspects_json, submitted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            rows_to_insert
        )

        taken = {r["username"] for r in db.execute("SELECT username FROM users").fetchall()}
        created = []
        for name in sorted(lecturer_names):
            username = _lecturer_username(name, taken)
            taken.add(username)
            db.execute(
                "INSERT INTO users (username, password_hash, role, lecturer_name, full_name) VALUES (?, ?, ?, ?, ?)",
                (username, generate_password_hash("lecturer123"), "lecturer", name, name)
            )
            created.append((username, name))

        db.execute(
            "INSERT INTO site_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("nigerian_live_imported", "true")
        )
        db.commit()
        print(
            f"[INFO] Imported {len(rows_to_insert)} Nigerian live evaluations and {len(created)} lecturer accounts.",
            flush=True
        )
        return len(rows_to_insert)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        raise
    finally:
        if owns_connection:
            db.close()

def get_site_settings():
    """Retrieves dynamic landing page and portal settings from the database."""
    settings = dict(DEFAULT_SETTINGS)
    try:
        db = get_db()
        rows = db.execute('SELECT key, value FROM site_settings').fetchall()
        for r in rows:
            settings[r['key']] = r['value']
    except Exception:
        pass
    return settings

def update_site_settings(new_settings):
    """Updates dynamic landing page settings in the database."""
    db = get_db()
    for key, val in new_settings.items():
        db.execute(
            'INSERT INTO site_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value',
            (key, str(val))
        )
    db.commit()


