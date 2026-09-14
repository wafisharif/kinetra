from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

import requests

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


# ---------------------------------------------------------------------------
# AI Coach — server-side Claude proxy.
#
# Design choice, deliberate: the API key lives ONLY here, as an environment
# variable on this server, never in the mobile app. The app never talks to
# Anthropic directly. That means:
#   1. The key can never leak by someone decompiling the app or sniffing its
#      network traffic.
#   2. This is the one place cost, rate limits, and what data leaves the
#      device are actually enforced.
#
# Data-minimization, also deliberate: this endpoint accepts ONLY numeric
# scores, letter grades, task names, and small trend arrays of past scores
# for the same task -- never video, never raw pose landmarks, never a photo.
# That is a real privacy boundary, not just a comment: everything this
# endpoint accepts is validated by type below, so there is no field a client
# could smuggle a video frame or free-text health note through.
#
# Fails closed, not open: if ANTHROPIC_API_KEY is not set, this returns a
# clean 503 rather than crashing the whole server or silently no-opping.
# That means this feature can ship in the app today and simply stay off
# until you add the key to your host's config -- no code change needed to
# turn it on later.
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_TIMEOUT_SECONDS = int(os.environ.get("ANTHROPIC_TIMEOUT_SECONDS", "20"))

# Separate, tighter rate limit from /analyze -- each call here costs real
# money, unlike a local video analysis.
AI_COACH_RATE_LIMIT_MAX_REQUESTS = int(
    os.environ.get("AI_COACH_RATE_LIMIT_MAX_REQUESTS", "8"))
AI_COACH_RATE_LIMIT_WINDOW_SECONDS = int(
    os.environ.get("AI_COACH_RATE_LIMIT_WINDOW_SECONDS", "600"))
_ai_coach_request_log = defaultdict(deque)


def _is_ai_coach_rate_limited(client_key: str) -> bool:
    now = time.time()
    q = _ai_coach_request_log[client_key]
    while q and now - q[0] > AI_COACH_RATE_LIMIT_WINDOW_SECONDS:
        q.popleft()
    if len(q) >= AI_COACH_RATE_LIMIT_MAX_REQUESTS:
        return True
    q.append(now)
    return False


_VALID_GRADE_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -/.'"
)


def _clean_short_string(value, max_len=60):
    """Coerce to a short, plain string or None. Strips anything that isn't
    a simple grade/label character so nothing free-form (an attempted
    prompt injection, a pasted note) reaches the model as if it were a
    trusted field."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = "".join(ch for ch in text if ch in _VALID_GRADE_CHARS)
    return text[:max_len] if text else None


def _clean_score(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score or score in (float("inf"), float("-inf")):  # NaN/inf
        return None
    return max(0, min(100, round(score)))


@app.route("/ai-coach", methods=["POST"])
def ai_coach():
    if not ANTHROPIC_API_KEY:
        return jsonify({
            "error": "AI Coach not configured",
            "details": (
                "This server does not have an ANTHROPIC_API_KEY set, so AI "
                "Coach is turned off. Nothing was sent anywhere."
            ),
        }), 503

    if _is_ai_coach_rate_limited(_client_key()):
        return jsonify({
            "error": "Too many requests",
            "details": (
                f"Limit is {AI_COACH_RATE_LIMIT_MAX_REQUESTS} AI Coach "
                f"requests per {AI_COACH_RATE_LIMIT_WINDOW_SECONDS} seconds. "
                "Try again shortly."
            ),
        }), 429

    payload = request.get_json(silent=True) or {}

    task_label = _clean_short_string(payload.get("task_label")) or "Movement Check"
    mode = _clean_short_string(payload.get("mode"), max_len=20) or "daily"
    primary_grade = _clean_short_string(payload.get("primary_grade")) or "N/A"
    confidence_grade = _clean_short_string(payload.get("confidence_grade"))
    side = _clean_short_string(payload.get("side"), max_len=10)
    primary_score = _clean_score(payload.get("primary_score"))
    thresholds_calibrated = bool(payload.get("thresholds_calibrated"))

    raw_recent_scores = payload.get("recent_scores")
    recent_scores = []
    if isinstance(raw_recent_scores, list):
        for item in raw_recent_scores[:10]:  # hard cap, ignore the rest
            cleaned = _clean_score(item)
            if cleaned is not None:
                recent_scores.append(cleaned)

    facts_lines = [
        f"- Task: {task_label} (mode: {mode})",
        f"- Most recent score: {primary_score if primary_score is not None else 'N/A'}/100",
        f"- Most recent grade: {primary_grade}",
    ]
    if confidence_grade:
        facts_lines.append(f"- Tracking confidence for this recording: {confidence_grade}")
    if side:
        facts_lines.append(f"- Side tested: {side}")
    facts_lines.append(
        f"- Personalized calibration active: {'yes' if thresholds_calibrated else 'no (using default range)'}"
    )
    if recent_scores:
        facts_lines.append(
            f"- Recent scores for this same task, oldest to newest: {', '.join(str(s) for s in recent_scores)}"
        )

    facts_block = "\n".join(facts_lines)

    system_prompt = (
        "You are a movement-coaching assistant inside a phone app called Kinetra. "
        "The app scores camera-based movement checks (built on Google MediaPipe "
        "pose estimation). You are given ONLY numeric scores and letter grades for "
        "one task -- you have not seen any video, image, or raw pose data, and you "
        "must never claim otherwise.\n\n"
        "Hard rules, no exceptions:\n"
        "1. You are not a doctor, physical therapist, or athletic trainer, and you "
        "must never diagnose a condition, name a suspected injury, or claim a "
        "medical cause for a score. If a score suggests something concerning, say "
        "so in plain terms and recommend an in-person clinician or athletic "
        "trainer -- do not guess what might be wrong.\n"
        "2. Keep the response to 3-5 short sentences. No headers, no bullet "
        "lists, no markdown -- this renders as plain text in a mobile app.\n"
        "3. Be specific to the numbers given, not generic fitness advice. If "
        "there's a trend in recent scores, mention it plainly.\n"
        "4. Never invent data you were not given (no fake percentages, no claims "
        "about joints or muscles not implied by the task name).\n"
        "5. Warm, direct, plain language -- talking to a teenager or an adult "
        "rehab patient, not a clinician."
    )

    user_prompt = (
        f"Here is this person's most recent Kinetra result:\n{facts_block}\n\n"
        "Write a short, encouraging, specific coaching note about this result."
    )

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 300,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=ANTHROPIC_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        return jsonify({
            "error": "AI Coach timed out",
            "details": "The AI coaching service took too long to respond. Try again shortly.",
        }), 504
    except requests.exceptions.RequestException as e:
        print(f"AI Coach network error: {e}")
        return jsonify({
            "error": "AI Coach unavailable",
            "details": "Could not reach the AI coaching service. Try again shortly.",
        }), 502

    if response.status_code != 200:
        print(f"AI Coach upstream error {response.status_code}: {response.text[:500]}")
        return jsonify({
            "error": "AI Coach unavailable",
            "details": "The AI coaching service returned an error. Try again shortly.",
        }), 502

    try:
        data = response.json()
        feedback_text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    except (ValueError, AttributeError) as e:
        print(f"AI Coach response parse error: {e}")
        feedback_text = ""

    if not feedback_text:
        return jsonify({
            "error": "AI Coach unavailable",
            "details": "The AI coaching service returned an empty response. Try again shortly.",
        }), 502

    return jsonify({
        "feedback": feedback_text,
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
