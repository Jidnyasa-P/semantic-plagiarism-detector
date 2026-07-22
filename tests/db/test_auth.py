import sqlite3
import uuid

import pytest

from src.db.auth import (
    add_user,
    delete_user,
    disable_2fa,
    enable_2fa,
    get_2fa_status,
    get_user_role,
    init_db,
    update_password,
    verify_user,
)


@pytest.fixture(autouse=True)
def db_connection():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute(
        """
                CREATE TABLE IF NOT EXISTS users (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT    UNIQUE NOT NULL,
                    password TEXT    NOT NULL,
                    role     TEXT    NOT NULL DEFAULT 'teacher',
                    tour_completed INTEGER DEFAULT 0,
                    otp_secret TEXT DEFAULT NULL,
                    two_factor_enabled INTEGER DEFAULT 0
                )
            """
    )
    conn.commit()
    yield conn
    print("In-memory database ready for testing")
    conn.close()


# Calls the init_db function and then uses verify_user to check if default admin user created
def test_init_db():
    init_db()
    assert verify_user("admin", "admin123") is not False


# Adds new user via uuid and uses get_user_role to check if user added
def test_add_user():
    user = uuid.uuid4().hex
    add_user(user, "ac_123")
    check = get_user_role(user)
    assert check is not None


# Adds a user and then checks whether adding same user again raises exception
def test_duplicate_user():
    add_user("hnsdf9", "ehns-1")
    with pytest.raises(sqlite3.IntegrityError):
        add_user("hnsdf9", "ehns-1")


# Checks whether adding incorrect password returns False
def test_verify_user():
    assert verify_user("hnsdf9", "ehns-1") is True
    assert verify_user("hnsdf9", "ehns_1") is False


def test_get_user_role():
    assert get_user_role("hnsdf9") is not None
    assert get_user_role("sdgk") is None


def test_update_password():
    update_password("hnsdf9", "sfgxv")
    assert verify_user("hnsdf9", "sfgxv") is not False


# Deletes a user and then verifies if it still exists
# No need to change the username as for each run since del is last operation and
# duplicate_user first it gets created and deleted for each run
def test_delete_user():
    delete_user("hnsdf9")
    assert get_user_role("hnsdf9") is None


def test_2fa_flow():
    username = "test2fauser"
    add_user(username, "pass123")

    enabled, secret = get_2fa_status(username)
    assert enabled is False
    assert secret is None

    test_secret = "JBSWY3DPEHPK3PXP"
    enable_2fa(username, test_secret)

    enabled, secret = get_2fa_status(username)
    assert enabled is True
    assert secret == test_secret

    disable_2fa(username)

    enabled, secret = get_2fa_status(username)
    assert enabled is False
    assert secret is None

    delete_user(username)
