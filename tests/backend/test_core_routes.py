"""Smoke tests for the routes that exist independent of AI Coach."""
import io
import os
from unittest.mock import patch


def test_root_route_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_health_route_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_analyze_with_no_video_is_a_clean_400_not_a_crash(client):
    # No file attached at all -- this should be a normal validation error,
    # never a 500. A 500 here would mean an unhandled exception in request
    # parsing, which is exactly the kind of thing that's easy to introduce
    # by accident while touching the /analyze route.
    resp = client.post("/analyze", data={})
    assert resp.status_code in (400, 422)


def test_rate_limiting_keys_on_the_proxy_forwarded_ip_not_one_shared_value(client, server_module):
    # End-to-end regression test for the ProxyFix hardening. The Flask test
    # client always connects from the same loopback socket address, so
    # before ProxyFix was wired in, _client_key() would have returned that
    # same value for every request regardless of X-Forwarded-For -- meaning
    # two different real-world clients behind the same proxy would have
    # silently shared one rate-limit bucket. Exhaust the limit for one
    # forwarded IP and confirm a different forwarded IP is unaffected.
    limit = server_module.RATE_LIMIT_MAX_REQUESTS
    for _ in range(limit):
        resp = client.post(
            "/analyze", data={}, headers={"X-Forwarded-For": "203.0.113.7"}
        )
        # Not rate-limited yet -- just the normal "no video uploaded" error.
        assert resp.status_code == 400

    blocked = client.post(
        "/analyze", data={}, headers={"X-Forwarded-For": "203.0.113.7"}
    )
    assert blocked.status_code == 429

    unaffected = client.post(
        "/analyze", data={}, headers={"X-Forwarded-For": "198.51.100.9"}
    )
    assert unaffected.status_code == 400


def test_analyze_uses_unique_per_request_csv_and_cleans_it_up(client):
    # Regression test: the debug CSV output path used to be a single shared
    # "elbow_angles.csv" for every request, which is a latent race condition
    # if this is ever deployed with more than one worker. It should now be
    # unique per request and deleted once the request finishes, same as the
    # uploaded video.
    captured = {}

    def fake_run_analysis(**kwargs):
        csv_path = kwargs["output_csv"]
        captured["output_csv"] = csv_path
        with open(csv_path, "w") as f:
            f.write("frame,angle\n")
        return {"total_frames": 1, "movement_health_score": 90}

    with patch("server.run_analysis", side_effect=fake_run_analysis):
        resp = client.post(
            "/analyze",
            data={"video": (io.BytesIO(b"fake video bytes"), "clip.mp4")},
            content_type="multipart/form-data",
        )

    assert resp.status_code == 200
    csv_path = captured["output_csv"]
    assert "elbow_angles.csv" not in csv_path
    assert csv_path.endswith(".csv")
    assert not os.path.exists(csv_path)


def test_analyze_gives_each_request_a_different_csv_path(client):
    seen_paths = []

    def fake_run_analysis(**kwargs):
        seen_paths.append(kwargs["output_csv"])
        return {"total_frames": 1, "movement_health_score": 90}

    with patch("server.run_analysis", side_effect=fake_run_analysis):
        for _ in range(2):
            client.post(
                "/analyze",
                data={"video": (io.BytesIO(b"fake video bytes"), "clip.mp4")},
                content_type="multipart/form-data",
            )

    assert len(seen_paths) == 2
    assert seen_paths[0] != seen_paths[1]
