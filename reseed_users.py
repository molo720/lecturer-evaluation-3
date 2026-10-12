#!/usr/bin/env python3
"""
Force reseed users table to ensure default accounts exist.
Run this script locally to fix login issues on Railway.
"""
import os
import sqlite3
from werkzeug.security import generate_password_hash

def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def data_dir():
    """Resolve data/ vs Data/"""
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

def reseed_users():
    """Drop and recreate users table with default accounts."""
    print("Reseeding users table...")
    print(f"Database path: {DATABASE_PATH}")
    
    # Initialize database tables
    db = sqlite3.connect(DATABASE_PATH, timeout=30.0)
    db.row_factory = sqlite3.Row
    
    # Create users table if not exists
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
    
    # Delete existing users
    db.execute('DELETE FROM users')
    print("Cleared existing users")
    
    # Re-insert default users
    for username, password, role, lecturer_name, full_name in DEMO_USERS:
        db.execute(
            'INSERT INTO users (username, password_hash, role, lecturer_name, full_name) VALUES (?, ?, ?, ?, ?)',
            (username, generate_password_hash(password), role, lecturer_name, full_name)
        )
        print(f"  Created user: {username} ({role})")
    
    db.commit()
    
    # Verify
    count = db.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    print(f"\nTotal users in database: {count}")
    
    # List all users
    print("\nCurrent users:")
    for row in db.execute('SELECT username, role, lecturer_name FROM users'):
        print(f"  - {row['username']} ({row['role']}) - {row['lecturer_name']}")
    
    print("\nDefault credentials:")
    print("  Admin: admin / admin123")
    print("  Lecturers: okafor, adeyemi, martins, faith, pomele, balogun, chukwu, adeleke")
    print("  Password for all lecturers: lecturer123")
    
    db.close()
    print("\n✅ Users reseeded successfully!")
    print(f"\nDatabase location: {DATABASE_PATH}")

if __name__ == "__main__":
    reseed_users()
