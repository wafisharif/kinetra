import os
import tempfile
import unittest

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-unittest-only")


class SyncTestCase(unittest.TestCase):
    def setUp(self):
        # Fresh DB file per test so tests never interfere with each other --
        # same pattern as test_accounts.py, and deliberately the *same*
        # ACCOUNTS_DB_PATH env var, since accounts.py and sync.py share one
        # SQLite file (sync.py imports DB_PATH from accounts.py).
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.remove(path)
        os.environ["ACCOUNTS_DB_PATH"] = path
        self.db_path = path

        import importlib
        import accounts as accounts_module
        importlib.reload(accounts_module)
        self.accounts = accounts_module
        self.accounts.init_db()

        import sync as sync_module
        importlib.reload(sync_module)
        self.sync = sync_module
        self.sync.init_sync_db()

        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(self.accounts.accounts_bp)
        app.register_blueprint(self.sync.sync_bp)
        self.client = app.test_client()

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def signup_and_get_token(self, email="sync@example.com", password="correcthorse"):
        resp = self.client.post(
            "/auth/signup",
            json={"name": "Sync Test", "email": email, "password": password},
        )
        return resp.get_json()["token"]

    def auth_headers(self, token):
        return {"Authorization": f"Bearer {token}"}

    def sample_payload(self):
        return {
            "sessions": [{"id": "s1", "timestamp": "2026-01-01T00:00:00.000Z", "primary_score": 80}],
            "checkins": [{"id": "c1", "timestamp": "2026-01-01T00:00:00.000Z", "feeling": 4}],
            "calibration": {"left": {"flex": 30, "extend": 150}, "right": None},
        }

    # --- push (/sync POST) --------------------------------------------

    def test_push_requires_auth(self):
        resp = self.client.post("/sync", json=self.sample_payload())
        self.assertEqual(resp.status_code, 401)

    def test_push_rejects_garbage_token(self):
        resp = self.client.post(
            "/sync",
            json=self.sample_payload(),
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        self.assertEqual(resp.status_code, 401)

    def test_push_success(self):
        token = self.signup_and_get_token()
        resp = self.client.post("/sync", json=self.sample_payload(), headers=self.auth_headers(token))
        self.assertEqual(resp.status_code, 200)
        self.assertIn("synced_at", resp.get_json())

    def test_push_rejects_non_object_body(self):
        token = self.signup_and_get_token()
        resp = self.client.post("/sync", json=[1, 2, 3], headers=self.auth_headers(token))
        self.assertEqual(resp.status_code, 400)

    def test_push_rejects_sessions_not_a_list(self):
        token = self.signup_and_get_token()
        payload = self.sample_payload()
        payload["sessions"] = "not-a-list"
        resp = self.client.post("/sync", json=payload, headers=self.auth_headers(token))
        self.assertEqual(resp.status_code, 400)
        self.assertIn("sessions", resp.get_json()["error"])

    def test_push_rejects_checkins_not_a_list(self):
        token = self.signup_and_get_token()
        payload = self.sample_payload()
        payload["checkins"] = {"oops": True}
        resp = self.client.post("/sync", json=payload, headers=self.auth_headers(token))
        self.assertEqual(resp.status_code, 400)

    def test_push_rejects_calibration_not_an_object(self):
        token = self.signup_and_get_token()
        payload = self.sample_payload()
        payload["calibration"] = "nope"
        resp = self.client.post("/sync", json=payload, headers=self.auth_headers(token))
        self.assertEqual(resp.status_code, 400)

    def test_push_rejects_too_many_sessions(self):
        token = self.signup_and_get_token()
        payload = self.sample_payload()
        payload["sessions"] = [{"id": str(i)} for i in range(self.sync.MAX_SESSIONS + 1)]
        resp = self.client.post("/sync", json=payload, headers=self.auth_headers(token))
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Too many sessions", resp.get_json()["error"])

    def test_push_rejects_too_many_checkins(self):
        token = self.signup_and_get_token()
        payload = self.sample_payload()
        payload["checkins"] = [{"id": str(i)} for i in range(self.sync.MAX_CHECKINS + 1)]
        resp = self.client.post("/sync", json=payload, headers=self.auth_headers(token))
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Too many check-ins", resp.get_json()["error"])

    def test_push_rejects_oversized_payload(self):
        token = self.signup_and_get_token()
        payload = self.sample_payload()
        # One giant note field, well under MAX_SESSIONS but over the byte cap.
        payload["sessions"] = [{"id": "s1", "note": "x" * (self.sync.MAX_PAYLOAD_BYTES + 1)}]
        resp = self.client.post("/sync", json=payload, headers=self.auth_headers(token))
        self.assertEqual(resp.status_code, 400)
        self.assertIn("too large", resp.get_json()["error"])

    def test_push_allows_null_calibration(self):
        token = self.signup_and_get_token()
        payload = self.sample_payload()
        payload["calibration"] = None
        resp = self.client.post("/sync", json=payload, headers=self.auth_headers(token))
        self.assertEqual(resp.status_code, 200)

    def test_push_is_full_replace_not_merge(self):
        token = self.signup_and_get_token()
        first = self.sample_payload()
        self.client.post("/sync", json=first, headers=self.auth_headers(token))

        second = {"sessions": [], "checkins": [], "calibration": {}}
        self.client.post("/sync", json=second, headers=self.auth_headers(token))

        resp = self.client.get("/sync", headers=self.auth_headers(token))
        data = resp.get_json()
        self.assertEqual(data["sessions"], [])
        self.assertEqual(data["checkins"], [])
        self.assertEqual(data["calibration"], {})

    def test_push_rate_limiting(self):
        token = self.signup_and_get_token()
        for _ in range(self.sync.SYNC_RATE_LIMIT_MAX_REQUESTS):
            self.client.post("/sync", json=self.sample_payload(), headers=self.auth_headers(token))
        resp = self.client.post("/sync", json=self.sample_payload(), headers=self.auth_headers(token))
        self.assertEqual(resp.status_code, 429)

    # --- pull (/sync GET) ------------------------------------------------

    def test_pull_requires_auth(self):
        resp = self.client.get("/sync")
        self.assertEqual(resp.status_code, 401)

    def test_pull_with_no_data_yet_returns_empty_defaults(self):
        token = self.signup_and_get_token(email="empty@example.com")
        resp = self.client.get("/sync", headers=self.auth_headers(token))
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["sessions"], [])
        self.assertEqual(data["checkins"], [])
        self.assertEqual(data["calibration"], {})
        self.assertIsNone(data["synced_at"])

    def test_pull_returns_previously_pushed_data(self):
        token = self.signup_and_get_token(email="roundtrip@example.com")
        payload = self.sample_payload()
        self.client.post("/sync", json=payload, headers=self.auth_headers(token))

        resp = self.client.get("/sync", headers=self.auth_headers(token))
        data = resp.get_json()
        self.assertEqual(data["sessions"], payload["sessions"])
        self.assertEqual(data["checkins"], payload["checkins"])
        self.assertEqual(data["calibration"], payload["calibration"])
        self.assertIsNotNone(data["synced_at"])

    def test_pull_is_scoped_to_the_authenticated_user(self):
        token_a = self.signup_and_get_token(email="a@example.com")
        token_b = self.signup_and_get_token(email="b@example.com")

        payload_a = self.sample_payload()
        payload_a["sessions"][0]["id"] = "only-for-a"
        self.client.post("/sync", json=payload_a, headers=self.auth_headers(token_a))

        resp_b = self.client.get("/sync", headers=self.auth_headers(token_b))
        self.assertEqual(resp_b.get_json()["sessions"], [])

        resp_a = self.client.get("/sync", headers=self.auth_headers(token_a))
        self.assertEqual(resp_a.get_json()["sessions"][0]["id"], "only-for-a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
