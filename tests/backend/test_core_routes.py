"""Smoke tests for the routes that exist independent of AI Coach."""


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
