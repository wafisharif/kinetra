# ---------------------------------------------------------------------------
# Web dashboard data sync.
#
# This is what makes the numbers on kinetraapp.com/dashboard real instead of
# generated sample data. The mobile app has no account system of its own
# (see accounts.py and the Privacy Policy) -- everything it records lives
# only in on-device AsyncStorage. This module is the one place that data is
# ever allowed to leave the device: when someone signs in to the *same*
# dashboard account from within the app (Settings -> Web Dashboard Sync),
# the app pushes a copy of its local sessions/check-ins/calibration here so
# the browser dashboard has something real to show. Nothing is ever synced
# without that explicit sign-in.
#
# Sync strategy is deliberately "full replace", not merge/diff: every push
# overwrites the entire stored row for that user with whatever the app sent.
# For a fundamentally single-user, effectively-single-device personal
# tracking app, this is enough -- it avoids a whole class of conflict-
# resolution bugs (interleaved edits, out-of-order pushes, partial merges)
# that would matter for a real multi-device sync product but don't earn
# their complexity here. The on-device data is always the source of truth;
# this table is a mirror of "whatever the app last pushed," not an
# independent store the app ever reads back into itself.
#
# Kept in its own module, alongside accounts.py, for the same reason that
# one is separate from pose_test.py: importable and testable without
# mediapipe/opencv/numpy in the path. Reuses accounts.py's token
# verification directly rather than re-implementing auth -- a sync request
# is authenticated exactly like every other dashboard request.
#
# Storage note: same SQLite file/ephemeral-disk caveat as accounts.py (see
# that module's header comment). A synced_data row disappears on the same
# redeploys that would also wipe the users table, so the two are at least
# consistently stale together.
# ---------------------------------------------------------------------------

import json
import os
import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path

from flask import Blueprint, request, jsonify

from accounts import verify_token, get_user_by_id, DB_PATH

sync_bp = Blueprint("sync", __name__)

# Generous but bounded -- enough for years of real daily use, small enough
# that a misbehaving or malicious client can't use this endpoint to store
# arbitrary large blobs. The app itself already caps saved sessions to the
# most recent 30 (see saveAnalysisSession in the app), so a legitimate
# payload is always far under these limits.
MAX_SESSIONS = 500
MAX_CHECKINS = 1000
MAX_PAYLOAD_BYTES = 500_000  # 500 KB serialized


def init_sync_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_data (
                user_id TEXT PRIMARY KEY,
                sessions TEXT NOT NULL,
                checkins TEXT NOT NULL,
                calibration TEXT NOT NULL,
                synced_at REAL NOT NULL
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


class SyncError(Exception):
    """Raised for any expected, client-facing sync validation failure.
    Carries its own HTTP status so the route can stay thin, matching
    AccountError's shape in accounts.py."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def _validate_payload(payload):
    sessions = payload.get("sessions")
    checkins = payload.get("checkins")
    calibration = payload.get("calibration")

    if not isinstance(sessions, list):
        raise SyncError("'sessions' must be a list.")
    if not isinstance(checkins, list):
        raise SyncError("'checkins' must be a list.")
    if calibration is not None and not isinstance(calibration, dict):
        raise SyncError("'calibration' must be an object.")

    if len(sessions) > MAX_SESSIONS:
        raise SyncError(f"Too many sessions (max {MAX_SESSIONS}).")
    if len(checkins) > MAX_CHECKINS:
        raise SyncError(f"Too many check-ins (max {MAX_CHECKINS}).")

    calibration = calibration or {}

    # Measure the size of what will actually be stored, not the raw request
    # body -- catches an oversized payload even if Content-Length lied.
    serialized = json.dumps(
        {"sessions": sessions, "checkins": checkins, "calibration": calibration}
    )
    if len(serialized.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise SyncError(f"Payload too large (max {MAX_PAYLOAD_BYTES} bytes).")

    return sessions, checkins, calibration


def save_sync_data(user_id, sessions, checkins, calibration):
    synced_at = time.time()
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO sync_data (user_id, sessions, checkins, calibration, synced_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                sessions = excluded.sessions,
                checkins = excluded.checkins,
                calibration = excluded.calibration,
                synced_at = excluded.synced_at
            """,
            (
                user_id,
                json.dumps(sessions),
                json.dumps(checkins),
                json.dumps(calibration),
                synced_at,
            ),
        )
        conn.commit()
        return synced_at
    finally:
        conn.close()


def load_sync_data(user_id):
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM sync_data WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row:
            return {
                "sessions": [],
                "checkins": [],
                "calibration": {},
                "synced_at": None,
            }
        return {
            "sessions": json.loads(row["sessions"]),
            "checkins": json.loads(row["checkins"]),
            "calibration": json.loads(row["calibration"]),
            "synced_at": row["synced_at"],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Rate limiting -- same in-memory sliding-window approach as accounts.py's
# auth endpoints and /analyze elsewhere in this app. Sync isn't a brute-force
# target the way login is, but a signed-in client's own bugs (a retry loop,
# a sync-on-every-keystroke mistake) could still hammer this endpoint, so it
# gets its own, more generous counters.
# ---------------------------------------------------------------------------
SYNC_RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("SYNC_RATE_LIMIT_MAX_REQUESTS", "60"))
SYNC_RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("SYNC_RATE_LIMIT_WINDOW_SECONDS", "600"))
_sync_request_log = defaultdict(deque)


def _is_sync_rate_limited(client_key):
    now = time.time()
    q = _sync_request_log[client_key]
    while q and now - q[0] > SYNC_RATE_LIMIT_WINDOW_SECONDS:
        q.popleft()
    if len(q) >= SYNC_RATE_LIMIT_MAX_REQUESTS:
        return True
    q.append(now)
    return False


def _bearer_token():
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    return header[len("Bearer "):].strip() or None


def _authenticate_request():
    """Returns a user dict for a valid Bearer token, or raises SyncError
    with a 401 status. Shared by both routes below so the auth failure
    modes (missing header, expired/garbage token, deleted account) are
    handled identically in one place."""
    token = _bearer_token()
    if not token:
        raise SyncError("Not signed in", 401)

    user_id = verify_token(token)
    if not user_id:
        raise SyncError("Session expired or invalid. Please sign in again.", 401)

    user = get_user_by_id(user_id)
    if not user:
        raise SyncError("Account no longer exists.", 401)

    return user


@sync_bp.route("/sync", methods=["POST"])
def push_sync():
    try:
        user = _authenticate_request()

        if _is_sync_rate_limited(user["id"]):
            raise SyncError(
                f"Too many sync requests. Limit is {SYNC_RATE_LIMIT_MAX_REQUESTS} "
                f"per {SYNC_RATE_LIMIT_WINDOW_SECONDS} seconds. Try again shortly.",
                429,
            )

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise SyncError("Request body must be a JSON object.")

        sessions, checkins, calibration = _validate_payload(payload)
    except SyncError as e:
        return jsonify({"error": e.message}), e.status

    synced_at = save_sync_data(user["id"], sessions, checkins, calibration)
    return jsonify({"synced_at": synced_at}), 200


@sync_bp.route("/sync", methods=["GET"])
def pull_sync():
    try:
        user = _authenticate_request()
    except SyncError as e:
        return jsonify({"error": e.message}), e.status

    data = load_sync_data(user["id"])
    return jsonify(data), 200
