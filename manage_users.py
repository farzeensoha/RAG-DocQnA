"""
Standalone helper for provisioning accounts — run from the command line,
not exposed through the app itself.

Usage:
    python manage_users.py add <username> <password> <admin|user>
    python manage_users.py list
"""
import sys
from db import get_conn, hash_password, init_db


def add_user(username: str, password: str, role: str):
    if role not in ("admin", "user"):
        print("Role must be 'admin' or 'user'.")
        return
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, hash_password(password), role),
        )
        conn.commit()
        print(f"Added {role} account '{username}'.")
    except Exception as e:
        print(f"Could not add user: {e}")
    finally:
        conn.close()


def list_users():
    conn = get_conn()
    rows = conn.execute("SELECT id, username, role FROM users ORDER BY id").fetchall()
    conn.close()
    for r in rows:
        print(f"{r['id']:>3}  {r['username']:<20} {r['role']}")


if __name__ == "__main__":
    init_db()
    if len(sys.argv) == 2 and sys.argv[1] == "list":
        list_users()
    elif len(sys.argv) == 5 and sys.argv[1] == "add":
        add_user(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(__doc__)
