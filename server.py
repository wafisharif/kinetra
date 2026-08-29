from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import json
import sys
import uuid
from pathlib import Path

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
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    if not POSE_SCRIPT.exists():
        return jsonify({
            "error": "Backend misconfigured",
            "details": "pose_test.py was not found on the server."
        }), 500

    file = request.files["video"]
    mode = request.form.get("mode", "rep")
    daily_task = request.form.get("daily_task", "reach")

    if file.filename == "":
        return jsonify({
            "error": "Empty upload",
            "details": "No video filename was provided."
        }), 400

    upload_id = uuid.uuid4().hex
    filepath = UPLOAD_DIR / f"uploaded_{upload_id}.mp4"

    try:
        file.save(filepath)

        env = os.environ.copy()
        env["BACKEND_MODE"] = "1"
        env["VIDEO_PATH"] = str(filepath)
        env["MODE"] = mode
        env["DAILY_TASK"] = daily_task

        result = subprocess.run(
            [sys.executable, str(POSE_SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(BASE_DIR),
            timeout=ANALYSIS_TIMEOUT_SECONDS
        )

        print("\n=== ANALYSIS REQUEST ===")
        print(f"mode={mode}")
        print(f"daily_task={daily_task}")
        print(f"video={filepath}")

        print("\n=== STDOUT ===")
        print(result.stdout)

        print("\n=== STDERR ===")
        print(result.stderr)

        if result.returncode != 0:
            return jsonify({
                "error": "Analysis script failed",
                "details": f"pose_test.py exited with code {result.returncode}",
                "stderr": result.stderr,
                "raw_output": result.stdout
            }), 500

        output = result.stdout.strip()

        if not output:
            return jsonify({
                "error": "Empty analysis output",
                "details": "pose_test.py did not return JSON output.",
                "stderr": result.stderr
            }), 500

        try:
            parsed = json.loads(output)
        except Exception as e:
            return jsonify({
                "error": "Processing failed",
                "details": str(e),
                "stderr": result.stderr,
                "raw_output": result.stdout
            }), 500

        return jsonify(parsed), 200

    except subprocess.TimeoutExpired:
        return jsonify({
            "error": "Analysis timeout",
            "details": f"Analysis took longer than {ANALYSIS_TIMEOUT_SECONDS} seconds. Try a shorter video."
        }), 504

    except Exception as e:
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
