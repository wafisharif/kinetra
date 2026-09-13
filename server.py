from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from pose_test import run_analysis

app = Flask(__name__)
CORS(app)

# Keep uploads reasonable for deployment.
# 80 MB is still large, but safer than unlimited uploads.
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

POSE_SCRIPT = BASE_DIR / "pose_test.py"
ANALYSIS_TIMEOUT_SECONDS = int(
    os.environ.get("ANALYSIS_TIMEOUT_SECONDS", "90"))

# ---------------------------------------------------------------------------
# Cross-platform hard timeout for the analysis call.
#
# run_analysis() now runs IN this process (see the note at the top of
# pose_test.py for why), so we can no longer rely on subprocess.run(...,
# timeout=...) to kill a stuck analysis. signal.SIGALRM gives us the same
# guarantee on the platform that actually matters for this: the Linux dyno
# this is deployed on (Procfile runs a single sync gunicorn worker, so the
# request is always handled on that worker's main thread -- exactly where
# SIGALRM works). SIGALRM does not exist on Windows, so when this is run
# locally for development (`python server.py` on Windows), the timeout is
# simply not enforced instead of crashing -- your dev videos are short
# anyway, and the deployed server is what actually needs the guarantee.
# ---------------------------------------------------------------------------
import signal  # noqa: E402  (kept near where it's used)


def _run_with_timeout(func, timeout_seconds, *args, **kwargs):
    if not hasattr(signal, "SIGALRM"):
        return func(*args, **kwargs)

    def _on_alarm(signum, frame):
        raise TimeoutError(
            f"Analysis took longer than {timeout_seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout_seconds)
    try:
        return func(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


# ---------------------------------------------------------------------------
# Minimal best-effort rate limiting.
#
# This is NOT hardened security (a client can spoof X-Forwarded-For), but
# /analyze currently has no auth at all and runs a real video-processing
# pipeline per request -- this is just a cheap guard against one script
# flooding the single worker with requests, not a defense against a
# determined attacker. Revisit with real auth before this is used beyond
# your own testing/beta group.
# ---------------------------------------------------------------------------
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(
    os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "600"))
_request_log = defaultdict(deque)


def _is_rate_limited(client_key: str) -> bool:
    now = time.time()
    q = _request_log[client_key]
    while q and now - q[0] > RATE_LIMIT_WINDOW_SECONDS:
        q.popleft()
    if len(q) >= RATE_LIMIT_MAX_REQUESTS:
        return True
    q.append(now)
    return False


def _client_key() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "ok",
        "service": "kinetra-analysis-backend",
        "message": "Backend root is reachable. Use /health for health checks and /analyze for analysis."
    }), 200


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "service": "kinetra-analysis-backend",
        "message": "Backend is reachable",
        "pose_script_exists": POSE_SCRIPT.exists(),
        "analysis_timeout_seconds": ANALYSIS_TIMEOUT_SECONDS
    }), 200


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "File too large",
        "details": "The uploaded video is too large. Record a shorter video and try again."
    }), 413


@app.route("/analyze", methods=["POST"])
def analyze():
    if _is_rate_limited(_client_key()):
        return jsonify({
            "error": "Too many requests",
            "details": f"Limit is {RATE_LIMIT_MAX_REQUESTS} analyses per {RATE_LIMIT_WINDOW_SECONDS} seconds. Try again shortly."
        }), 429

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["video"]
    mode = request.form.get("mode", "rep")
    daily_task = request.form.get("daily_task", "reach")
    # Optional Phase 1 additions. Both default to exactly what run_analysis()
    # itself defaults to when a field is entirely absent from the request,
    # so an older/unmodified frontend client that never sends these gets
    # byte-identical behavior to before they existed. run_analysis() does
    # its own validation and safe fallback for malformed values (an
    # unrecognized side, a non-numeric or degenerate threshold pair), so
    # raw form values are passed through as-is rather than re-validated here.
    side = request.form.get("side", "right")
    flex_threshold = request.form.get("flex_threshold")
    extend_threshold = request.form.get("extend_threshold")

    if file.filename == "":
        return jsonify({
            "error": "Empty upload",
            "details": "No video filename was provided."
        }), 400

    upload_id = uuid.uuid4().hex
    filepath = UPLOAD_DIR / f"uploaded_{upload_id}.mp4"

    try:
        file.save(filepath)

        print("\n=== ANALYSIS REQUEST ===")
        print(f"mode={mode}")
        print(f"daily_task={daily_task}")
        print(f"side={side}")
        print(f"flex_threshold={flex_threshold} extend_threshold={extend_threshold}")
        print(f"video={filepath}")

        result_data = _run_with_timeout(
            run_analysis,
            ANALYSIS_TIMEOUT_SECONDS,
            video_path=str(filepath),
            mode=mode,
            daily_task=daily_task,
            output_csv=str(BASE_DIR / "elbow_angles.csv"),
            backend_mode=True,
            side=side,
            flex_threshold=flex_threshold,
            extend_threshold=extend_threshold,
        )

        print("=== ANALYSIS OK ===")
        print(f"total_frames={result_data.get('total_frames')} "
              f"movement_health_score={result_data.get('movement_health_score')}")

        return jsonify(result_data), 200

    except TimeoutError:
        return jsonify({
            "error": "Analysis timeout",
            "details": f"Analysis took longer than {ANALYSIS_TIMEOUT_SECONDS} seconds. Try a shorter video."
        }), 504

    except RuntimeError as e:
        # Raised by run_analysis() itself, e.g. an unreadable/corrupt video.
        return jsonify({
            "error": "Analysis failed",
            "details": str(e)
        }), 500

    except Exception as e:
        print(f"Unexpected error during analysis: {e}")
        return jsonify({
            "error": "Server error",
            "details": str(e)
        }), 500

    finally:
        try:
            if filepath.exists():
                filepath.unlink()
        except Exception as cleanup_error:
            print(f"Cleanup failed for {filepath}: {cleanup_error}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
