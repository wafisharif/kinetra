"""Shared pytest fixtures for the Flask backend test suite.

server.py imports `run_analysis` from pose_test at module level, so
importing server anywhere (even for tests that never touch /analyze)
requires the full requirements.txt (mediapipe, opencv, ...) to be
installed. That's intentional -- it's the same import graph the real
deployed server has, so a green test run here is a real signal that the
backend actually boots, not just that isolated functions work.
"""
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest


@pytest.fixture
def server_module(monkeypatch):
    """Import (or re-import) server.py with a clean, known environment.

    Reloading per-test keeps module-level state (rate-limit queues, the
    ANTHROPIC_API_KEY read at import time) from leaking between tests --
    each test starts from the same blank slate.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AI_COACH_RATE_LIMIT_MAX_REQUESTS", raising=False)
    monkeypatch.delenv("AI_COACH_RATE_LIMIT_WINDOW_SECONDS", raising=False)
    monkeypatch.delenv("RATE_LIMIT_MAX_REQUESTS", raising=False)
    monkeypatch.delenv("RATE_LIMIT_WINDOW_SECONDS", raising=False)

    if "server" in sys.modules:
        module = importlib.reload(sys.modules["server"])
    else:
        module = importlib.import_module("server")
    return module


@pytest.fixture
def client(server_module):
    return server_module.app.test_client()
