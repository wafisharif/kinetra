import os
import tempfile
import time
import unittest

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-unittest-only")


class AccountsTestCase(unittest.TestCase):
    def setUp(self):
        # Fresh DB file per test so tests never interfere with each other.
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

        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(self.accounts.accounts_bp)
        self.client = app.test_client()

    def tearDown(self):
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def signup(self, name="Jordan Test", email="jordan@example.com", password="correcthorse"):
        return self.client.post("/auth/signup", json={"name": name, "email": email, "password": password})

    def test_signup_success(self):
        resp = self.signup()
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()
        self.assertIn("token", data)
        self.assertEqual(data["user"]["name"], "Jordan Test")
        self.assertEqual(data["user"]["email"], "jordan@example.com")
        self.assertNotIn("password", data["user"])
        self.assertNotIn("password_hash", data["user"])

    def test_signup_rejects_duplicate_email_case_insensitive(self):
        self.signup(email="dup@example.com")
        resp = self.signup(email="DUP@example.com", name="Someone Else")
        self.assertEqual(resp.status_code, 409)
        self.assertIn("already exists", resp.get_json()["error"])

    def test_signup_rejects_short_password(self):
        resp = self.signup(password="short")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Password", resp.get_json()["error"])

    def test_signup_rejects_bad_email(self):
        resp = self.signup(email="not-an-email")
        self.assertEqual(resp.status_code, 400)

    def test_signup_rejects_empty_name(self):
        resp = self.signup(name="   ")
        self.assertEqual(resp.status_code, 400)

    def test_password_is_hashed_not_stored_plain(self):
        self.signup(email="hash@example.com", password="correcthorse")
        conn = __import__("sqlite3").connect(self.db_path)
        row = conn.execute("SELECT password_hash FROM users WHERE email_lower = ?", ("hash@example.com",)).fetchone()
        conn.close()
        self.assertNotEqual(row[0], "correcthorse")
        self.assertTrue(row[0].startswith("pbkdf2:") or ":" in row[0])

    def test_login_success(self):
        self.signup(email="login@example.com", password="correcthorse")
        resp = self.client.post("/auth/login", json={"email": "login@example.com", "password": "correcthorse"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("token", data)
        self.assertEqual(data["user"]["email"], "login@example.com")

    def test_login_wrong_password(self):
        self.signup(email="wrongpw@example.com", password="correcthorse")
        resp = self.client.post("/auth/login", json={"email": "wrongpw@example.com", "password": "nope"})
        self.assertEqual(resp.status_code, 401)

    def test_login_unknown_email_same_error_as_wrong_password(self):
        resp1 = self.client.post("/auth/login", json={"email": "ghost@example.com", "password": "whatever1"})
        self.signup(email="real@example.com", password="correcthorse")
        resp2 = self.client.post("/auth/login", json={"email": "real@example.com", "password": "wrongpass"})
        self.assertEqual(resp1.status_code, resp2.status_code)
        self.assertEqual(resp1.get_json()["error"], resp2.get_json()["error"])

    def test_login_is_case_insensitive_on_email(self):
        self.signup(email="Case@Example.com", password="correcthorse")
        resp = self.client.post("/auth/login", json={"email": "case@example.com", "password": "correcthorse"})
        self.assertEqual(resp.status_code, 200)

    def test_me_with_valid_token(self):
        signup_resp = self.signup(email="me@example.com")
        token = signup_resp.get_json()["token"]
        resp = self.client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["user"]["email"], "me@example.com")

    def test_me_without_token(self):
        resp = self.client.get("/auth/me")
        self.assertEqual(resp.status_code, 401)

    def test_me_with_garbage_token(self):
        resp = self.client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
        self.assertEqual(resp.status_code, 401)

    def test_me_with_tampered_token(self):
        signup_resp = self.signup(email="tamper@example.com")
        token = signup_resp.get_json()["token"]
        tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
        resp = self.client.get("/auth/me", headers={"Authorization": f"Bearer {tampered}"})
        self.assertEqual(resp.status_code, 401)

    def test_signup_rate_limiting(self):
        for i in range(self.accounts.AUTH_RATE_LIMIT_MAX_REQUESTS):
            self.signup(email=f"rl{i}@example.com")
        resp = self.signup(email="rl-overflow@example.com")
        self.assertEqual(resp.status_code, 429)

    def test_token_survives_round_trip_and_encodes_no_pii(self):
        signup_resp = self.signup(email="pii@example.com", name="Secret Name")
        token = signup_resp.get_json()["token"]
        # Token should not contain the raw email or name in plain text.
        self.assertNotIn("pii@example.com", token)
        self.assertNotIn("Secret", token)
        user_id = self.accounts.verify_token(token)
        self.assertIsNotNone(user_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
