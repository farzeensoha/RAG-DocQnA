import os
import sqlite3
import hashlib
import secrets

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def hash_password(password: str, salt: str = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return secrets.compare_digest(check, digest)


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'user'))
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_by INTEGER REFERENCES users(id),
            uploaded_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS qa_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id INTEGER REFERENCES users(id),
            model_used TEXT NOT NULL,
            prompt TEXT NOT NULL,
            output TEXT NOT NULL,
            reviewer_status TEXT NOT NULL DEFAULT 'none'
                CHECK(reviewer_status IN ('none', 'pending_review', 'reviewed')),
            reviewed_by INTEGER REFERENCES users(id),
            reviewed_at TEXT,
            reviewer_notes TEXT
        );
        """
    )
    conn.commit()

    # Seed accounts only once, from env vars, so credentials are never hardcoded.
    existing = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    if existing == 0:
        admin_user = os.getenv("SEED_ADMIN_USERNAME", "admin")
        admin_pass = os.getenv("SEED_ADMIN_PASSWORD", "changeme-admin")
        emp_user = os.getenv("SEED_USER_USERNAME", "employee")
        emp_pass = os.getenv("SEED_USER_PASSWORD", "changeme-user")

        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'admin')",
            (admin_user, hash_password(admin_pass)),
        )
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'user')",
            (emp_user, hash_password(emp_pass)),
        )
        conn.commit()
        print(
            f"[db] Seeded initial accounts — admin: '{admin_user}', user: '{emp_user}'. "
            "Set SEED_ADMIN_USERNAME/SEED_ADMIN_PASSWORD/SEED_USER_USERNAME/SEED_USER_PASSWORD "
            "in .env before first run to control these, or add more accounts with manage_users.py."
        )

    conn.close()


def get_user_by_username(username: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row


def get_user_by_id(user_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row
