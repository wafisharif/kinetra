"""Route-level tests for /analyze-hand (Phase 1).

test_hand_tasks.py already covers hand_tasks.run_hand_analysis() itself in
isolation. This file covers the NEW wiring in server.py: request parsing,
the HAND_TASKS validation, and that _execute_analysis_and_respond (shared
with /analyze) produces the same status codes/cleanup behavior for this
route as it already does for /analyze -- see test_core_routes.py for the
equivalent tests on that route, which this file deliberately mirrors.
"""
import io
import os
from unittest.mock import patch

from recording_quality import RecordingQualityError


def test_analyze_hand_with_no_video_is_a_clean_400(client):
    resp = client.post("/analyze-hand", data={})
    assert resp.status_code == 400


def test_analyze_hand_rejects_unknown_hand_task_with_400_not_500(client):
    resp = client.post(
        "/analyze-hand",
        data={
            "video": (io.BytesIO(b"fake video bytes"), "clip.mp4"),
            "hand_task": "not_a_real_task",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "hand_task" in resp.get_json()["details"]


def test_analyze_hand_happy_path_passes_fields_through_and_cleans_up(client):
    captured = {}

    def fake_run_hand_analysis(**kwargs):
        captured.update(kwargs)
        return {
            "hand_task": kwargs["hand_task"],
            "hand": kwargs["hand"],
            "trial_index": kwargs["trial_index"],
            "total_frames": 90,
            "task_analysis": {"task_type": kwargs["hand_task"], "tap_count": 12},
        }

    with patch("server.run_hand_analysis", side_effect=fake_run_hand_analysis):
        resp = client.post(
            "/analyze-hand",
            data={
                "video": (io.BytesIO(b"fake video bytes"), "clip.mp4"),
                "hand_task": "finger_tapping_camera",
                "hand": "left",
                "trial_index": "2",
            },
            content_type="multipart/form-data",
        )

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["hand_task"] == "finger_tapping_camera"
    assert body["hand"] == "left"
    assert body["trial_index"] == 2  # parsed from the string form field

    assert captured["hand_task"] == "finger_tapping_camera"
    assert captured["hand"] == "left"
    assert captured["trial_index"] == 2
    # Uploaded video path was passed through, and cleaned up afterward --
    # same contract as /analyze's own uploaded-file cleanup.
    uploaded_path = captured["video_path"]
    assert not os.path.exists(uploaded_path)


def test_analyze_hand_defaults_hand_task_and_hand_when_omitted(client):
    captured = {}

    def fake_run_hand_analysis(**kwargs):
        captured.update(kwargs)
        return {"hand_task": kwargs["hand_task"], "hand": kwargs["hand"]}

    with patch("server.run_hand_analysis", side_effect=fake_run_hand_analysis):
        resp = client.post(
            "/analyze-hand",
            data={"video": (io.BytesIO(b"fake video bytes"), "clip.mp4")},
            content_type="multipart/form-data",
        )

    assert resp.status_code == 200
    assert captured["hand_task"] == "finger_tapping_camera"
    assert captured["hand"] == "right"
    assert captured["trial_index"] is None


def test_analyze_hand_surfaces_recording_quality_error_as_422(client):
    def fake_run_hand_analysis(**kwargs):
        raise RecordingQualityError(["Hand wasn't visible for enough of this recording."])

    with patch("server.run_hand_analysis", side_effect=fake_run_hand_analysis):
        resp = client.post(
            "/analyze-hand",
            data={"video": (io.BytesIO(b"fake video bytes"), "clip.mp4")},
            content_type="multipart/form-data",
        )

    assert resp.status_code == 422
    body = resp.get_json()
    assert "reasons" in body
    assert any("visible" in r for r in body["reasons"])


def test_analyze_hand_shares_the_same_rate_limit_bucket_as_analyze(client, server_module):
    # /analyze and /analyze-hand both call _is_rate_limited(_client_key()),
    # which reads/writes the same module-level _request_log -- deliberately
    # a single shared request budget per client, not two independent ones.
    # Exhausting it via /analyze should also block /analyze-hand.
    limit = server_module.RATE_LIMIT_MAX_REQUESTS
    headers = {"X-Forwarded-For": "203.0.113.55"}
    for _ in range(limit):
        resp = client.post("/analyze", data={}, headers=headers)
        assert resp.status_code == 400  # not rate-limited yet

    blocked = client.post(
        "/analyze-hand",
        data={"video": (io.BytesIO(b"fake video bytes"), "clip.mp4")},
        content_type="multipart/form-data",
        headers=headers,
    )
    assert blocked.status_code == 429
