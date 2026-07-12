"""
auth.py
-------
SQLite-backed authentication with bcrypt password hashing.

Public API
----------
init_db()                          → create tables + seed default admin
verify_user(username, password)    → bool
get_user_role(username)            → str | None
add_user(username, password, role) → None
get_all_users()                    → list[dict]
delete_user(username)              → None
update_password(username, password)→ None
"""

import sqlite3
import bcrypt
import os

_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "users.db"))


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH, check_same_thread=False)


def init_db() -> None:
    """Create users table and seed default admin if not exists."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT    UNIQUE NOT NULL,
                password TEXT    NOT NULL,
                role     TEXT    NOT NULL DEFAULT 'teacher'
            )
        """)
        conn.commit()
        exists = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", ("admin",)
        ).fetchone()
        if not exists:
            hashed = bcrypt.hashpw(b"admin123", bcrypt.gensalt(10)).decode()
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ("admin", hashed, "admin"),
            )
        conn.commit()


def verify_user(username: str, password: str) -> bool:
    """Return True if username exists and password matches the stored hash."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username.lower(),)
        ).fetchone()
    if not row:
        return False
    return bcrypt.checkpw(password.encode(), row[0].encode())


def get_user_role(username: str) -> str | None:
    """Return the role of a user, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT role FROM users WHERE username = ?", (username.lower(),)
        ).fetchone()
    return row[0] if row else None


def add_user(username: str, password: str, role: str = "teacher") -> None:
    """Insert a new user with a bcrypt-hashed password."""
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt(10)).decode()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username.lower(), hashed, role),
        )
        conn.commit()


def get_all_users() -> list:
    """Return all users as a list of dicts (excludes password hashes)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, username, role FROM users ORDER BY id"
        ).fetchall()
    return [{"id": r[0], "username": r[1], "role": r[2]} for r in rows]


def delete_user(username: str) -> None:
    """Delete a user by username."""
    with _connect() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username.lower(),))
        conn.commit()


def update_password(username: str, new_password: str) -> None:
    """Update a user's password with a new bcrypt hash."""
    hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt(10)).decode()
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET password = ? WHERE username = ?",
            (hashed, username.lower()),
        )
        conn.commit()
