# ---------------------------------------------------------------------------
# Web dashboard accounts.
#
# This is a real, separate account system for the browser dashboard at
# kinetraapp.com/dashboard. It has nothing to do with the mobile app, which
# still has no accounts and keeps everything on-device (see the Privacy
# Policy). Kept in its own module, importable and testable without pulling
# in pose_test.py's heavy dependencies (mediapipe, opencv, numpy), so the
# account system can be exercised by a plain Flask test client.
#
# Passwords are hashed with werkzeug's generate_password_hash/
# check_password_hash (PBKDF2, salted) -- the plain password is never
# stored, logged, or recoverable. Sessions are stateless signed tokens
# (itsdangerous.URLSafeTimedSerializer), not server-side session rows, so
# there is no session table to manage or expire by hand; a token simply
# stops verifying once it's older than TOKEN_MAX_AGE_SECONDS.
#
# Storage is SQLite in a single file. On this deployment (Render, one
# gunicorn worker, no persistent disk on the free tier) that file lives on
# the container's local, EPHEMERAL disk: it survives restarts of the same
# running instance, but a new deploy gets a fresh, empty database. That's a
# real limitation worth knowing about, not a secret one -- accounts here
# will not survive a redeploy until either a Render persistent disk is
# added or this is pointed at an external database via ACCOUNTS_DB_PATH
# (or a future swap to a hosted Postgres). Shipping this now, with the
# limitation flagged, was the deliberate choice over not shipping real
# accounts at all.
# ---------------------------------------------------------------------------

import os
import re
import sqlite3
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from flask import Blueprint, request, jsonify, g
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("ACCOUNTS_DB_PATH", str(BASE_DIR / "accounts.db")))

# Falls back to a random key generated at process start if SECRET_KEY isn't
# set, so the account system works out of the box in local dev. On a real
# deployment this means every worker restart invalidates every existing
# token (everyone gets signed out) until a real SECRET_KEY env var is set --
# the same fail-safe-not-fail-silent pattern as ANTHROPIC_API_KEY elsewhere
# in this app. Set SECRET_KEY on Render to make sign-ins durable across
# restarts (separately from the accounts-table durability note above).
SECRET_KEY = os.environ.get("SECRET_KEY", "").strip() or uuid.uuid4().hex
if not os.environ.get("SECRET_KEY", "").strip():
    print(
        "WARNING: SECRET_KEY is not set. Using a random key for this process "
        "only -- every dashboard sign-in will be invalidated on the next "
        "restart. Set a real SECRET_KEY on your host to fix this."
    )

_serializer = URLSafeTimedSerializer(SECRET_KEY, salt="kinetra-dashboard-auth")
TOKEN_MAX_AGE_SECONDS = 60 * 60 * 24 * 30  # 30 days

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8
MAX_NAME_LENGTH = 80
MAX_EMAIL_LENGTH = 254

accounts_bp = Blueprint("accounts", __name__)


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                email_lower TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


class AccountError(Exception):
    """Raised for any expected, user-facing signup/login failure. Carries
    its own HTTP status so routes can stay thin."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def _validate_name(name):
    name = (name or "").strip()
    if not name:
        raise AccountError("Enter your name.")
    if len(name) > MAX_NAME_LENGTH:
        raise AccountError(f"Name must be {MAX_NAME_LENGTH} characters or fewer.")
    return name


def _validate_email(email):
    email = (email or "").strip()
    if not email or len(email) > MAX_EMAIL_LENGTH or not EMAIL_RE.match(email):
        raise AccountError("Enter a valid email address.")
    return email


def _validate_password(password):
    password = password or ""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AccountError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        )
    if len(password) > 200:
        raise AccountError("Password is too long.")
    return password


def _user_public(row):
    return {"id": row["id"], "name": row["name"], "email": row["email"]}


def create_user(name, email, password):
    name = _validate_name(name)
    email = _validate_email(email)
    password = _validate_password(password)

    conn = _get_conn()
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE email_lower = ?", (email.lower(),)
        ).fetchone()
        if existing:
            raise AccountError("An account with that email already exists.", 409)

        user_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO users (id, name, email, email_lower, password_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, name, email, email.lower(), generate_password_hash(password), time.time()),
        )
        conn.commit()
        return {"id": user_id, "name": name, "email": email}
    finally:
        conn.close()


def authenticate_user(email, password):
    email = (email or "").strip()
    password = password or ""
    if not email or not password:
        raise AccountError("Enter your email and password.")

    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE email_lower = ?", (email.lower(),)
        ).fetchone()
        # Deliberately identical error for "no such user" and "wrong
        # password" -- distinguishing them lets an attacker enumerate which
        # emails have accounts.
        if not row or not check_password_hash(row["password_hash"], password):
            raise AccountError("Incorrect email or password.", 401)
        return _user_public(row)
    finally:
        conn.close()


def get_user_by_id(user_id):
    conn = _get_conn()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _user_public(row) if row else None
    finally:
        conn.close()


def generate_token(user_id):
    return _serializer.dumps({"uid": user_id})


def verify_token(token):
    try:
        data = _serializer.loads(token, max_age=TOKEN_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
    return data.get("uid")


# ---------------------------------------------------------------------------
# Rate limiting -- same in-memory sliding-window approach used for /analyze
# and /ai-coach elsewhere in this app. Signup/login are classic brute-force
# and account-enumeration targets, so they get their own counters, tighter
# than /analyze's.
# ---------------------------------------------------------------------------
AUTH_RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("AUTH_RATE_LIMIT_MAX_REQUESTS", "15"))
AUTH_RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("AUTH_RATE_LIMIT_WINDOW_SECONDS", "600"))
_auth_request_log = defaultdict(deque)


def _is_auth_rate_limited(client_key):
    now = time.time()
    q = _auth_request_log[client_key]
    while q and now - q[0] > AUTH_RATE_LIMIT_WINDOW_SECONDS:
        q.popleft()
    if len(q) >= AUTH_RATE_LIMIT_MAX_REQUESTS:
        return True
    q.append(now)
    return False


def _client_key():
    return request.remote_addr or "unknown"


def _bearer_token():
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    return header[len("Bearer "):].strip() or None


@accounts_bp.route("/auth/signup", methods=["POST"])
def signup():
    if _is_auth_rate_limited(_client_key()):
        return jsonify({
            "error": "Too many requests",
            "details": f"Limit is {AUTH_RATE_LIMIT_MAX_REQUESTS} attempts per {AUTH_RATE_LIMIT_WINDOW_SECONDS} seconds. Try again shortly.",
        }), 429

    payload = request.get_json(silent=True) or {}
    try:
        user = create_user(payload.get("name"), payload.get("email"), payload.get("password"))
    except AccountError as e:
        return jsonify({"error": e.message}), e.status

    token = generate_token(user["id"])
    return jsonify({"token": token, "user": user}), 201


@accounts_bp.route("/auth/login", methods=["POST"])
def login():
    if _is_auth_rate_limited(_client_key()):
        return jsonify({
            "error": "Too many requests",
            "details": f"Limit is {AUTH_RATE_LIMIT_MAX_REQUESTS} attempts per {AUTH_RATE_LIMIT_WINDOW_SECONDS} seconds. Try again shortly.",
        }), 429

    payload = request.get_json(silent=True) or {}
    try:
        user = authenticate_user(payload.get("email"), payload.get("password"))
    except AccountError as e:
        return jsonify({"error": e.message}), e.status

    token = generate_token(user["id"])
    return jsonify({"token": token, "user": user}), 200


@accounts_bp.route("/auth/me", methods=["GET"])
def me():
    token = _bearer_token()
    if not token:
        return jsonify({"error": "Not signed in"}), 401

    user_id = verify_token(token)
    if not user_id:
        return jsonify({"error": "Session expired or invalid. Please sign in again."}), 401

    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "Account no longer exists."}), 401

    return jsonify({"user": user}), 200
