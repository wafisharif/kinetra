import os
import json
import matplotlib.pyplot as plt
import cv2
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing
import numpy as np
import csv
from typing import Any, Dict

# -----------------------------
# CONFIG
# -----------------------------
BACKEND_MODE = os.getenv("BACKEND_MODE", "0") == "1"
video_path = os.getenv("VIDEO_PATH", "test_video.mp4")
output_csv = os.getenv("OUTPUT_CSV", "elbow_angles.csv")
MODE = os.getenv("MODE", "rep")
DAILY_TASK = os.getenv("DAILY_TASK", "reach")

pose = mp_pose.Pose()

# -----------------------------
# ANGLE CALCULATION
# -----------------------------


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return None

    cosine_angle = np.dot(ba, bc) / denominator
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# -----------------------------
# SIMPLE TORSO LEAN ESTIMATE
# -----------------------------


def calculate_torso_lean(shoulder, hip):
    shoulder = np.array(shoulder)
    hip = np.array(hip)

    vector = shoulder - hip

    if np.linalg.norm(vector) == 0:
        return None

    vertical = np.array([0, -1])
    cosine_angle = np.dot(vector, vertical) / \
        (np.linalg.norm(vector) * np.linalg.norm(vertical))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


# -----------------------------
# SMOOTHING
# -----------------------------
def moving_average(data, window_size=5):
    smoothed = []
    for i in range(len(data)):
        window = data[max(0, i - window_size + 1):i + 1]
        window = [v for v in window if v is not None]

        if len(window) == 0:
            smoothed.append(None)
        else:
            smoothed.append(sum(window) / len(window))

    return smoothed


# -----------------------------
# STATE-BASED REP COUNTING
# -----------------------------
def count_reps_bidirectional(data, low_thresh=95, high_thresh=115):
    state = "unknown"
    transitions = 0

    for angle in data:
        if angle is None:
            continue

        if state == "unknown":
            if angle > high_thresh:
                state = "extended"
            elif angle < low_thresh:
                state = "flexed"

        elif state == "extended":
            if angle < low_thresh:
                state = "flexed"
                transitions += 1

        elif state == "flexed":
            if angle > high_thresh:
                state = "extended"
                transitions += 1

    reps = transitions // 2
    return transitions, reps


# -----------------------------
# SIGNAL QUALITY
# -----------------------------
def compute_signal_quality(angles):
    total_frames = len(angles)
    valid_frames = len([a for a in angles if a is not None])

    if total_frames == 0:
        score = 0
    else:
        score = valid_frames / total_frames

    if score >= 0.8:
        grade = "Good"
        message = "Movement signal quality is good."
    elif score >= 0.5:
        grade = "Moderate"
        message = "Some landmarks were missing. Keep the shoulder, elbow, and wrist clearly visible."
    else:
        grade = "Poor"
        message = "Movement signal quality is poor. Re-record with the full arm visible and a stable camera angle."

    return {
        "score": round(score, 2),
        "valid_frames": valid_frames,
        "total_frames": total_frames,
        "grade": grade,
        "message": message
    }

# -----------------------------
# QUALITY METRICS
# -----------------------------


def compute_smoothness(velocity):
    valid = [v for v in velocity if v is not None]
    if len(valid) < 5:
        return None
    return np.std(valid)


def compute_symmetry(velocity):
    positive = [v for v in velocity if v is not None and v > 0]
    negative = [abs(v) for v in velocity if v is not None and v < 0]

    if not positive or not negative:
        return None

    return np.mean(positive) / np.mean(negative)


def compute_control(velocity):
    valid = [v for v in velocity if v is not None]
    if len(valid) < 2:
        return None

    spikes = sum(1 for v in valid if abs(v) > 15)
    return spikes / len(valid)

# -----------------------------
# MECHANICAL EFFICIENCY
# -----------------------------


def compute_efficiency(smoothness, symmetry, control):
    if smoothness is None or symmetry is None or control is None:
        return None

    smooth_term = 1 / (1 + smoothness)

    sym_term = max(0, 1 - abs(symmetry - 1))
    ctrl_term = max(0, 1 - control)

    efficiency = (
        0.4 * smooth_term +
        0.3 * sym_term +
        0.3 * ctrl_term
    )

    return efficiency


# -----------------------------
# GRADING FUNCTIONS
# -----------------------------
def grade_smoothness(val):
    if val is None:
        return "N/A"
    if val < 1:
        return "Good"
    elif val < 2:
        return "Moderate"
    else:
        return "Poor"


def grade_symmetry(val):
    if val is None:
        return "N/A"
    if 0.8 <= val <= 1.2:
        return "Good"
    elif val > 1.2:
        return "Poor (extension too fast)"
    else:
        return "Poor (flexion too fast)"


def grade_control(val):
    if val is None:
        return "N/A"
    if val < 0.05:
        return "Excellent"
    elif val < 0.15:
        return "Good"
    else:
        return "Poor"


def grade_efficiency(val):
    if val is None:
        return "N/A"
    if val > 0.75:
        return "Highly Efficient"
    elif val > 0.55:
        return "Moderately Efficient"
    else:
        return "Inefficient"


# -----------------------------
# FATIGUE ANALYSIS
# -----------------------------
def analyze_fatigue(rep_metrics):
    if len(rep_metrics) < 2:
        return "Insufficient data"

    smoothness_vals = [
        r["smoothness"] for r in rep_metrics if r["smoothness"] is not None
    ]
    control_vals = [
        r["control"] for r in rep_metrics if r["control"] is not None
    ]

    if len(smoothness_vals) < 2 or len(control_vals) < 2:
        return "Insufficient data"

    smoothness_trend = smoothness_vals[-1] - smoothness_vals[0]
    control_trend = control_vals[-1] - control_vals[0]

    fatigue_flags = []

    if smoothness_trend > 0.5:
        fatigue_flags.append("movement getting less smooth")

    if control_trend > 0.05:
        fatigue_flags.append("control decreasing")

    if fatigue_flags:
        return "Fatigue detected: " + ", ".join(fatigue_flags)

    return "No clear fatigue trend"

# -----------------------------
# PERFORMANCE INTERPRETATION
# -----------------------------


def interpret_degradation(performance_drop):
    if performance_drop is None:
        return "Not enough data"

    # -----------------------------
    # PERFORMANCE DECLINE (fatigue)
    # -----------------------------
    if performance_drop > 20:
        return "Significant performance drop across reps"
    elif performance_drop > 10:
        return "Moderate fatigue detected"
    elif performance_drop > 3:
        return "Slight fatigue buildup"

    # -----------------------------
    # PERFORMANCE IMPROVEMENT
    # -----------------------------
    elif performance_drop < -20:
        return "Significant improvement across reps"
    elif performance_drop < -10:
        return "Moderate improvement across reps"
    elif performance_drop < -3:
        return "Slight improvement across reps"

    # -----------------------------
    # STABLE
    # -----------------------------
    else:
        return "Performance remained stable"

# -----------------------------
# KEY INSIGHT ENGINE
# -----------------------------


def build_key_insights(global_metrics, rep_analysis, performance_summary, mode="rep"):
    insights = []

    smooth_grade = global_metrics.get("smoothness_grade")
    sym_grade = global_metrics.get("symmetry_grade")
    control_grade = global_metrics.get("control_grade")
    efficiency_grade = global_metrics.get("efficiency_grade")

    # -----------------------------
    # SYMMETRY INSIGHT
    # -----------------------------
    if sym_grade == "Good":
        insights.append(
            "Concentric and eccentric phases are mechanically balanced.")
    elif sym_grade and "extension too fast" in sym_grade:
        insights.append(
            "Extension velocity exceeds flexion, indicating imbalance in force application.")
    elif sym_grade and "flexion too fast" in sym_grade:
        insights.append(
            "Flexion velocity exceeds extension, indicating asymmetrical loading.")

    # -----------------------------
    # SMOOTHNESS INSIGHT
    # -----------------------------
    if smooth_grade == "Good":
        insights.append(
            "Angular velocity profile is smooth and well-regulated.")
    elif smooth_grade == "Moderate":
        insights.append(
            "Moderate variability in angular velocity suggests inconsistent force output.")
    elif smooth_grade == "Poor":
        insights.append(
            "High variability in angular velocity indicates unstable movement execution.")

    # -----------------------------
    # CONTROL INSIGHT
    # -----------------------------
    if control_grade == "Excellent":
        insights.append(
            "Minimal high-frequency fluctuations, indicating strong motor control.")
    elif control_grade == "Good":
        insights.append(
            "Some velocity spikes present, but overall control remains stable.")
    elif control_grade == "Poor":
        insights.append(
            "Frequent velocity spikes indicate poor control and instability.")

    # -----------------------------
    # EFFICIENCY INSIGHT
    # -----------------------------
    if efficiency_grade == "Highly Efficient":
        insights.append(
            "Energy transfer is efficient with minimal mechanical loss.")
    elif efficiency_grade == "Moderately Efficient":
        insights.append(
            "Moderate inefficiencies suggest suboptimal movement mechanics.")
    elif efficiency_grade == "Inefficient":
        insights.append(
            "Significant mechanical inefficiency suggests wasted motion or poor coordination.")

    # -----------------------------
    # PERFORMANCE TREND
    # -----------------------------
    if performance_summary and performance_summary != "Not enough data":
        insights.append(performance_summary + ".")

    # -----------------------------
    # BEST VS WORST MOVEMENT SEGMENT
    # -----------------------------
    scored_reps = [r for r in rep_analysis if r.get("score") is not None]

    if len(scored_reps) >= 2:
        best_rep = max(scored_reps, key=lambda x: x["score"])
        worst_rep = min(scored_reps, key=lambda x: x["score"])

        if best_rep["rep"] != worst_rep["rep"]:
            if mode == "daily":
                insights.append(
                    f"Movement was most stable during Cycle {best_rep['rep']} and least stable during Cycle {worst_rep['rep']}."
                )
            else:
                insights.append(
                    f"Peak performance occurred at Rep {best_rep['rep']}, with lowest performance at Rep {worst_rep['rep']}."
                )

    return insights

# -----------------------------
# MOVEMENT SIGNATURE
# -----------------------------


def classify_movement_signature(global_metrics, performance_summary, mode="rep"):
    smooth_grade = global_metrics.get("smoothness_grade")
    sym_grade = global_metrics.get("symmetry_grade")
    control_grade = global_metrics.get("control_grade")
    efficiency_grade = global_metrics.get("efficiency_grade")

    if performance_summary in [
        "Significant performance drop across reps",
        "Moderate fatigue detected",
        "Slight fatigue buildup"
    ]:
        if mode == "daily":
            return {
                "label": "Stability drop-off pattern",
                "description": "Movement quality decreased across cycles, suggesting reduced stability during repeated motion."
            }

        return {
            "label": "Fatigue-sensitive pattern",
            "description": "Movement quality decreases across reps, suggesting performance breakdown under repeated effort."
        }

    if performance_summary in [
        "Significant improvement across reps",
        "Moderate improvement across reps",
        "Slight improvement across reps"
    ]:
        if mode == "daily":
            return {
                "label": "Stability improvement pattern",
                "description": "Movement quality improved across cycles, suggesting the motion became more controlled after initial movement."
            }

        return {
            "label": "Warm-up responder",
            "description": "Movement quality improves across reps, suggesting the motion becomes more stable after initial repetitions."
        }

    if control_grade == "Poor":
        return {
            "label": "Control-limited pattern",
            "description": "Frequent velocity spikes suggest instability is the main limiter."
        }

    if sym_grade and "Poor" in sym_grade:
        return {
            "label": "Symmetry-limited pattern",
            "description": "Uneven flexion and extension speeds suggest asymmetrical movement mechanics."
        }

    if smooth_grade == "Poor":
        return {
            "label": "Smoothness-limited pattern",
            "description": "High angular velocity variability suggests inconsistent movement execution."
        }

    if efficiency_grade == "Inefficient":
        return {
            "label": "Efficiency-limited pattern",
            "description": "The movement shows signs of wasted motion or poor mechanical coordination."
        }

    return {
        "label": "Stable movement pattern",
        "description": "Movement mechanics are relatively consistent based on the current metrics."
    }

# -----------------------------
# INTERPRETATION LAYER
# -----------------------------


def interpret_results(mode, global_metrics, fatigue, daily_task="reach"):
    if mode == "rep":
        return {
            "summary": "Rep Quality Analysis",
            "focus": "Explosiveness, control, and rep-by-rep movement quality",
            "insight": fatigue
        }

    if mode == "rehab":
        return {
            "summary": "Rehab Consistency Review",
            "focus": "Stability, controlled motion, and repeatability",
            "insight": "Consistency matters more than speed in rehab-focused tracking"
        }

    if mode == "lab":
        return {
            "summary": "Movement Lab Analysis",
            "focus": "Signal behavior, symmetry, and detailed biomechanical trends",
            "insight": "Use the metrics and rep breakdown as raw movement-analysis outputs"
        }

    if mode == "daily":
        smooth_grade = global_metrics.get("smoothness_grade")
        control_grade = global_metrics.get("control_grade")
        symmetry_grade = global_metrics.get("symmetry_grade")
        efficiency_grade = global_metrics.get("efficiency_grade")
        task_label = get_daily_task_label(daily_task)

        daily_flags = []

        if smooth_grade == "Poor":
            daily_flags.append("movement was less smooth than ideal")
        if control_grade == "Poor":
            daily_flags.append("control was inconsistent")
        if symmetry_grade and "Poor" in symmetry_grade:
            daily_flags.append("movement was asymmetric")
        if efficiency_grade == "Inefficient":
            daily_flags.append("movement efficiency was reduced")

        if len(daily_flags) == 0:
            insight = "Movement appeared stable, controlled, and efficient during this check."
        else:
            insight = "Daily movement check found that " + \
                ", ".join(daily_flags) + "."

        return {
            "summary": f"Daily Movement Health: {task_label}",
            "focus": f"Everyday movement stability, control, symmetry, and efficiency during {task_label.lower()} motion",
            "insight": insight
        }

    return {
        "summary": "Movement Analysis",
        "focus": "General biomechanics review",
        "insight": fatigue
    }


# -----------------------------
# TRANSITIONS
# -----------------------------
def get_transition_indices(data, low_thresh=95, high_thresh=115):
    state = "unknown"
    indices = []

    for i, angle in enumerate(data):
        if angle is None:
            continue

        if state == "unknown":
            if angle > high_thresh:
                state = "extended"
            elif angle < low_thresh:
                state = "flexed"

        elif state == "extended":
            if angle < low_thresh:
                state = "flexed"
                indices.append(i)

        elif state == "flexed":
            if angle > high_thresh:
                state = "extended"
                indices.append(i)

    return indices


# -----------------------------
# SEGMENT REPS
# -----------------------------
def segment_reps(data, transition_indices):
    reps = []

    for i in range(0, len(transition_indices) - 1, 2):
        start = transition_indices[i]
        end = transition_indices[i + 1]
        segment = data[start:end]

        reps.append((start, end, segment))

    return reps


# -----------------------------
# ANALYZE REPS
# -----------------------------
def analyze_reps(reps, velocities):
    results = []

    for i, (start, end, segment) in enumerate(reps):
        vel_segment = velocities[start:end]

        smooth = compute_smoothness(vel_segment)
        sym = compute_symmetry(vel_segment)
        ctrl = compute_control(vel_segment)

        results.append({
            "rep": i + 1,
            "start": start,
            "end": end,
            "smoothness": smooth,
            "symmetry": sym,
            "control": ctrl
        })

    return results


# -----------------------------
# REP SCORING
# -----------------------------
def compute_rep_score(smoothness, symmetry, control):
    if smoothness is None or symmetry is None or control is None:
        return None

    smooth_score = max(0, 100 - smoothness * 5)
    sym_score = max(0, 100 - abs(symmetry - 1) * 100)
    control_score = max(0, 100 - control * 200)

    final_score = (
        0.4 * smooth_score +
        0.3 * sym_score +
        0.3 * control_score
    )

    return round(final_score, 1)

# -----------------------------
# MOVEMENT HEALTH SCORE
# -----------------------------


def compute_movement_health_score(global_metrics, performance_drop):
    smoothness = global_metrics.get("smoothness")
    symmetry = global_metrics.get("symmetry")
    control = global_metrics.get("control")
    efficiency = global_metrics.get("efficiency")

    if smoothness is None or symmetry is None or control is None or efficiency is None:
        return None

    smooth_score = max(0, 100 - smoothness * 5)
    symmetry_score = max(0, 100 - abs(symmetry - 1) * 100)
    control_score = max(0, 100 - control * 200)
    efficiency_score = max(0, min(100, efficiency * 100))

    if performance_drop is None:
        trend_score = 70
    elif performance_drop > 20:
        trend_score = 40
    elif performance_drop > 10:
        trend_score = 55
    elif performance_drop > 3:
        trend_score = 70
    elif performance_drop < -10:
        trend_score = 90
    elif performance_drop < -3:
        trend_score = 80
    else:
        trend_score = 85

    final_score = (
        0.25 * smooth_score +
        0.20 * symmetry_score +
        0.20 * control_score +
        0.25 * efficiency_score +
        0.10 * trend_score
    )

    return round(final_score, 1)


def grade_movement_health(score):
    if score is None:
        return "N/A"
    if score >= 85:
        return "Excellent Movement Health"
    elif score >= 70:
        return "Strong Movement Health"
    elif score >= 55:
        return "Moderate Movement Health"
    else:
        return "Reduced Movement Health"


def grade_score(score):
    if score is None:
        return "N/A"
    if score >= 85:
        return "Elite"
    elif score >= 70:
        return "Strong"
    elif score >= 55:
        return "Decent"
    else:
        return "Needs Work"

# -----------------------------
# SAFE JSON HELPERS
# -----------------------------

# -----------------------------
# DAILY TASK LABELS
# -----------------------------


def get_daily_task_label(task):
    if task == "reach":
        return "Reach"
    if task == "arm_raise":
        return "Arm Raise"
    if task == "sit_to_stand":
        return "Sit-to-Stand"
    if task == "walking":
        return "Walking"
    if task == "balance":
        return "Balance"
    if task == "timed_up_and_go":
        return "Timed Up and Go"
    return "General Movement"


def get_daily_task_focus(task):
    if task == "reach":
        return {
            "primary_goal": "Reach control and smoothness",
            "tracked_region": "Shoulder, elbow, and wrist",
            "current_status": "Currently analyzed using the elbow-based movement prototype.",
            "future_metrics": [
                "Reach path smoothness",
                "Endpoint steadiness",
                "Shoulder-elbow coordination",
                "Trajectory consistency"
            ]
        }

    if task == "arm_raise":
        return {
            "primary_goal": "Arm raise stability and coordination",
            "tracked_region": "Shoulder, elbow, and wrist",
            "current_status": "Currently analyzed using the elbow-based movement prototype.",
            "future_metrics": [
                "Shoulder range of motion",
                "Raising/lowering control",
                "Top-position steadiness",
                "Arm movement smoothness"
            ]
        }

    if task == "sit_to_stand":
        return {
            "primary_goal": "Sit-to-stand transition stability",
            "tracked_region": "Hip, knee, trunk, and lower body landmarks",
            "current_status": "Task selected, but true sit-to-stand mechanics are not implemented yet.",
            "future_metrics": [
                "Transition time",
                "Trunk lean",
                "Knee/hip coordination",
                "Rise stability",
                "Sit/stand phase detection"
            ]
        }

    if task == "walking":
        return {
            "primary_goal": "Walking rhythm and stability",
            "tracked_region": "Hips, knees, ankles, and feet",
            "current_status": "Walking V1 uses MediaPipe lower-body landmarks to estimate gait rhythm, step consistency, and walking stability.",
            "future_metrics": [
                "Step rhythm",
                "Gait path stability",
                "Left-right consistency",
                "Foot/ankle visibility",
                "YOLO-assisted person tracking"
            ]
        }

    if task == "balance":
        return {
            "primary_goal": "Standing balance and postural stability",
            "tracked_region": "Shoulders, hips, torso, ankles, and body center",
            "current_status": "Balance V1 uses MediaPipe body landmarks to estimate visible sway, torso control, and posture stability.",
            "future_metrics": [
                "Center sway",
                "Torso lean variability",
                "Hip drift",
                "Shoulder drift",
                "Static balance consistency",
                "Foot support stability"
            ]
        }

    if task == "timed_up_and_go":
        return {
            "primary_goal": "Functional mobility sequence quality",
            "tracked_region": "Full body, hips, knees, ankles, torso, and walking path",
            "current_status": "Timed Up and Go V1 estimates sit-to-stand, walking movement, turning/return pattern, and overall mobility sequence quality.",
            "future_metrics": [
                "Sit-to-stand phase timing",
                "Walking phase rhythm",
                "Turn detection",
                "Return path detection",
                "Sequence smoothness",
                "Functional mobility score"
            ]
        }

    return {
        "primary_goal": "General movement quality",
        "tracked_region": "Visible body landmarks",
        "current_status": "General movement check.",
        "future_metrics": [
            "Movement smoothness",
            "Control stability",
            "Symmetry",
            "Efficiency"
        ]
    }

# -----------------------------
# SIT-TO-STAND TASK SUMMARY
# -----------------------------


# -----------------------------
# SIT-TO-STAND TASK SUMMARY + TRANSITION DETECTION
# -----------------------------
def summarize_sit_to_stand_task(task_signal_data, total_frames, fps: float = 30.0):
    knee_vals = [
        v for v in task_signal_data["right_knee_angles"] if v is not None]
    hip_vals = [v for v in task_signal_data["right_hip_angles"] if v is not None]
    torso_vals = [
        v for v in task_signal_data["torso_lean_values"] if v is not None]

    valid_frames = task_signal_data["sit_to_stand_valid_frames"]

    if total_frames == 0:
        signal_score = 0
    else:
        signal_score = valid_frames / total_frames

    if signal_score >= 0.75:
        signal_grade = "Good"
    elif signal_score >= 0.45:
        signal_grade = "Moderate"
    else:
        signal_grade = "Poor"

    # If there is not enough lower-body data, return safely
    if len(knee_vals) < 5 or len(hip_vals) < 5:
        return {
            "task_type": "sit_to_stand",
            "status": "insufficient_lower_body_data",
            "right_knee_angle_mean": safe_float(np.mean(knee_vals)) if knee_vals else None,
            "right_hip_angle_mean": safe_float(np.mean(hip_vals)) if hip_vals else None,
            "torso_lean_mean": safe_float(np.mean(torso_vals)) if torso_vals else None,
            "lower_body_valid_frames": valid_frames,
            "lower_body_signal_score": round(signal_score, 2),
            "lower_body_signal_grade": signal_grade,
            "transition_detected": False,
            "transition_start_frame": None,
            "transition_end_frame": None,
            "transition_duration_sec": None,
            "knee_extension_range": None,
            "hip_extension_range": None,
            "max_torso_lean": safe_float(max(torso_vals)) if torso_vals else None,
            "torso_control": None,
            "rise_stability_score": None,
            "rise_stability_grade": "N/A",
            "summary": "Not enough lower-body landmark data to detect a sit-to-stand transition.",
            "note": "Re-record with the full body visible from the side, especially hips, knees, and ankles."
        }

    knee_array = np.array(knee_vals)
    hip_array = np.array(hip_vals)
    torso_array = np.array(torso_vals) if torso_vals else np.array([])

    # Smooth the lower-body signals lightly
    knee_smooth = moving_average(knee_array.tolist(), window_size=5)
    hip_smooth = moving_average(hip_array.tolist(), window_size=5)

    # Sit-to-stand usually causes knee angle to increase as the person stands
    knee_min = float(np.min(knee_smooth))
    knee_max = float(np.max(knee_smooth))
    hip_min = float(np.min(hip_smooth))
    hip_max = float(np.max(hip_smooth))

    knee_range = knee_max - knee_min
    hip_range = hip_max - hip_min

    # If the knee barely changes, no real sit-to-stand transition was detected
    if knee_range < 15:
        return {
            "task_type": "sit_to_stand",
            "status": "no_clear_transition",
            "right_knee_angle_mean": safe_float(np.mean(knee_vals)),
            "right_hip_angle_mean": safe_float(np.mean(hip_vals)),
            "torso_lean_mean": safe_float(np.mean(torso_vals)) if torso_vals else None,
            "lower_body_valid_frames": valid_frames,
            "lower_body_signal_score": round(signal_score, 2),
            "lower_body_signal_grade": signal_grade,
            "transition_detected": False,
            "transition_start_frame": None,
            "transition_end_frame": None,
            "transition_duration_sec": None,
            "knee_extension_range": safe_float(knee_range),
            "hip_extension_range": safe_float(hip_range),
            "max_torso_lean": safe_float(max(torso_vals)) if torso_vals else None,
            "torso_control": None,
            "rise_stability_score": None,
            "rise_stability_grade": "N/A",
            "summary": "No clear sit-to-stand transition was detected.",
            "note": "Try recording a full seated-to-standing motion from the side."
        }

    # Define transition start/end using percentage of knee extension range
    start_threshold = knee_min + 0.15 * knee_range
    end_threshold = knee_min + 0.85 * knee_range

    transition_start = None
    transition_end = None

    for i, val in enumerate(knee_smooth):
        if transition_start is None and val >= start_threshold:
            transition_start = i

        if transition_start is not None and val >= end_threshold:
            transition_end = i
            break

    if transition_start is None or transition_end is None or transition_end <= transition_start:
        transition_detected = False
        transition_duration = None
    else:
        transition_detected = True
        transition_duration = (transition_end - transition_start) / fps

    max_torso_lean = float(np.max(torso_array)) if len(
        torso_array) > 0 else None

    if (
        transition_detected
        and transition_start is not None
        and transition_end is not None
        and len(torso_array) > transition_end
    ):
        torso_segment = torso_array[transition_start:transition_end + 1]
        torso_control = float(np.std(torso_segment)) if len(
            torso_segment) > 1 else 0
    else:
        torso_control = None

    # Basic stability score
    if transition_detected:
        # Faster is not automatically better. We reward controlled, not chaotic.
        duration_penalty = 0

        if transition_duration is not None:
            if transition_duration < 0.5:
                duration_penalty = 15
            elif transition_duration > 4.0:
                duration_penalty = 10

        lean_penalty = 0
        if max_torso_lean is not None:
            lean_penalty = min(30, max(0, (max_torso_lean - 20) * 1.5))

        control_penalty = 0
        if torso_control is not None:
            control_penalty = min(25, torso_control * 2)

        signal_penalty = 0
        if signal_score < 0.75:
            signal_penalty = 20
        elif signal_score < 0.9:
            signal_penalty = 8

        rise_stability_score = max(
            0,
            100 - duration_penalty - lean_penalty - control_penalty - signal_penalty
        )

        rise_stability_score = round(rise_stability_score, 1)

        if rise_stability_score >= 85:
            rise_stability_grade = "Excellent"
        elif rise_stability_score >= 70:
            rise_stability_grade = "Good"
        elif rise_stability_score >= 55:
            rise_stability_grade = "Moderate"
        else:
            rise_stability_grade = "Needs Work"
    else:
        rise_stability_score = None
        rise_stability_grade = "N/A"

    if transition_detected:
        summary = "Sit-to-stand transition detected and analyzed."
        note = "This is an early sit-to-stand analysis using knee, hip, and torso motion."
    else:
        summary = "Sit-to-stand transition could not be reliably detected."
        note = "Try recording from the side with the full body visible."

    return {
        "task_type": "sit_to_stand",
        "status": "sit_to_stand_v1_analysis",
        "right_knee_angle_mean": safe_float(np.mean(knee_vals)),
        "right_hip_angle_mean": safe_float(np.mean(hip_vals)),
        "torso_lean_mean": safe_float(np.mean(torso_vals)) if torso_vals else None,
        "lower_body_valid_frames": valid_frames,
        "lower_body_signal_score": round(signal_score, 2),
        "lower_body_signal_grade": signal_grade,
        "transition_detected": transition_detected,
        "transition_start_frame": transition_start,
        "transition_end_frame": transition_end,
        "transition_duration_sec": safe_float(transition_duration),
        "knee_extension_range": safe_float(knee_range),
        "hip_extension_range": safe_float(hip_range),
        "max_torso_lean": safe_float(max_torso_lean),
        "torso_control": safe_float(torso_control),
        "rise_stability_score": rise_stability_score,
        "rise_stability_grade": rise_stability_grade,
        "summary": summary,
        "note": note
    }

# -----------------------------
# SIT-TO-STAND TASK INSIGHTS
# -----------------------------


def build_sit_to_stand_insights(task_analysis):
    insights = []

    if task_analysis is None:
        return ["Sit-to-stand analysis was not available for this recording."]

    if not task_analysis.get("transition_detected"):
        insights.append(
            "A clear sit-to-stand transition was not detected. Re-record from the side with hips, knees, and ankles visible."
        )

        if task_analysis.get("lower_body_signal_grade") == "Poor":
            insights.append(
                "Lower-body signal quality was poor, so the app could not reliably track the movement."
            )

        return insights

    transition_duration = task_analysis.get("transition_duration_sec")
    knee_range = task_analysis.get("knee_extension_range")
    hip_range = task_analysis.get("hip_extension_range")
    max_torso_lean = task_analysis.get("max_torso_lean")
    torso_control = task_analysis.get("torso_control")
    stability_score = task_analysis.get("rise_stability_score")
    signal_grade = task_analysis.get("lower_body_signal_grade")

    # Transition duration insight
    if transition_duration is not None:
        if transition_duration < 0.5:
            insights.append(
                "The sit-to-stand transition was very fast, which may indicate rushed or less controlled movement."
            )
        elif transition_duration > 4.0:
            insights.append(
                "The sit-to-stand transition was slow, which may suggest reduced power, hesitation, or cautious movement."
            )
        else:
            insights.append(
                "Transition timing was within a reasonable range for a controlled sit-to-stand."
            )

    # Knee range insight
    if knee_range is not None:
        if knee_range < 25:
            insights.append(
                "Knee extension range was limited, suggesting the standing motion may have been incomplete or poorly captured."
            )
        else:
            insights.append(
                "Knee extension changed clearly during the movement, supporting a valid sit-to-stand transition."
            )

    # Hip range insight
    if hip_range is not None:
        if hip_range < 20:
            insights.append(
                "Hip motion range was limited, which may mean the hips were not clearly visible or the transition was incomplete."
            )
        else:
            insights.append(
                "Hip motion changed meaningfully during the rise, which supports lower-body transition detection."
            )

    # Torso lean insight
    if max_torso_lean is not None:
        if max_torso_lean > 35:
            insights.append(
                "Torso lean was high during the rise, suggesting the movement relied heavily on forward trunk motion."
            )
        elif max_torso_lean > 20:
            insights.append(
                "Torso lean was moderate during the rise."
            )
        else:
            insights.append(
                "Torso lean stayed relatively controlled during the rise."
            )

    # Torso control insight
    if torso_control is not None:
        if torso_control > 8:
            insights.append(
                "Torso motion varied noticeably during the transition, suggesting reduced trunk control."
            )
        elif torso_control > 4:
            insights.append(
                "Torso control showed mild variability during the transition."
            )
        else:
            insights.append(
                "Torso control remained stable during the transition."
            )

    # Rise stability insight
    if stability_score is not None:
        if stability_score >= 85:
            insights.append(
                "Overall rise stability was excellent for this recording."
            )
        elif stability_score >= 70:
            insights.append(
                "Overall rise stability was good, with only minor movement-control issues."
            )
        elif stability_score >= 55:
            insights.append(
                "Overall rise stability was moderate and could be improved."
            )
        else:
            insights.append(
                "Overall rise stability needs work, likely due to trunk lean, control variability, or signal quality."
            )

    # Signal quality insight
    if signal_grade == "Poor":
        insights.append(
            "Because lower-body signal quality was poor, treat these sit-to-stand results as low-confidence."
        )
    elif signal_grade == "Moderate":
        insights.append(
            "Lower-body signal quality was moderate, so the result is usable but should be interpreted cautiously."
        )
    elif signal_grade == "Good":
        insights.append(
            "Lower-body signal quality was good, so the sit-to-stand metrics are more trustworthy."
        )

    return insights

# -----------------------------
# REACH TASK ANALYSIS
# -----------------------------


def summarize_reach_task(task_signal_data, total_frames):

    distances = task_signal_data["reach_distances"]

    if len(distances) < 5:
        return {
            "task_type": "reach",
            "status": "insufficient_reach_data"
        }

    reach_array = np.array(distances)

    reach_distance_max = float(np.max(reach_array))
    reach_distance_min = float(np.min(reach_array))

    reach_range = reach_distance_max - reach_distance_min

    reach_smoothness = float(np.std(np.diff(reach_array)))

    endpoint_window = reach_array[-10:] if len(
        reach_array) >= 10 else reach_array

    endpoint_steadiness = float(np.std(endpoint_window))

    valid_frames = task_signal_data["reach_valid_frames"]

    signal_score = valid_frames / max(total_frames, 1)

    reach_stability_score = 100

    reach_stability_score -= min(30, reach_smoothness * 40)
    reach_stability_score -= min(20, endpoint_steadiness * 50)

    if signal_score < 0.75:
        reach_stability_score -= 15

    reach_stability_score = max(0, round(reach_stability_score, 1))

    if reach_stability_score >= 85:
        grade = "Excellent"
    elif reach_stability_score >= 70:
        grade = "Good"
    elif reach_stability_score >= 55:
        grade = "Moderate"
    else:
        grade = "Needs Work"

    return {
        "task_type": "reach",
        "status": "reach_v1_analysis",
        "reach_distance_max": safe_float(reach_distance_max),
        "reach_distance_min": safe_float(reach_distance_min),
        "reach_range": safe_float(reach_range),
        "reach_smoothness": safe_float(reach_smoothness),
        "endpoint_steadiness": safe_float(endpoint_steadiness),
        "reach_stability_score": reach_stability_score,
        "reach_stability_grade": grade
    }


def build_reach_insights(task_analysis):

    insights = []

    if task_analysis is None:
        return ["Reach analysis unavailable."]

    reach_range = task_analysis.get("reach_range")
    smoothness = task_analysis.get("reach_smoothness")
    steadiness = task_analysis.get("endpoint_steadiness")
    score = task_analysis.get("reach_stability_score")

    if reach_range is not None:
        if reach_range < 0.15:
            insights.append(
                "Reach distance changed only slightly during the movement."
            )
        else:
            insights.append(
                "Reach distance changed clearly throughout the movement."
            )

    if smoothness is not None:
        if smoothness > 0.03:
            insights.append(
                "Reach motion showed noticeable variability and reduced smoothness."
            )
        else:
            insights.append(
                "Reach motion remained smooth and controlled."
            )

    if steadiness is not None:
        if steadiness > 0.03:
            insights.append(
                "The hand was less steady near the end of the reach."
            )
        else:
            insights.append(
                "The hand remained stable near the end of the reach."
            )

    if score is not None:
        insights.append(
            f"Overall reach stability score: {score}/100."
        )

    return insights

# -----------------------------
# ARM RAISE TASK ANALYSIS
# -----------------------------


def summarize_arm_raise_task(task_signal_data, total_frames):

    vertical_positions = task_signal_data["arm_raise_vertical_positions"]
    distances = task_signal_data["arm_raise_distances"]
    elbow_angles = [
        v for v in task_signal_data["arm_raise_elbow_angles"] if v is not None
    ]

    if len(vertical_positions) < 5:
        return {
            "task_type": "arm_raise",
            "status": "insufficient_arm_raise_data",
            "summary": "Not enough arm raise data was captured.",
            "note": "Re-record with the shoulder, elbow, and wrist visible."
        }

    vertical_array = np.array(vertical_positions)
    distance_array = np.array(distances)

    raise_max = float(np.max(vertical_array))
    raise_min = float(np.min(vertical_array))
    raise_range = raise_max - raise_min

    arm_raise_smoothness = float(np.std(np.diff(vertical_array)))

    # Look near the highest raised position, not just the final frames.
    top_index = int(np.argmax(vertical_array))
    start_idx = max(0, top_index - 5)
    end_idx = min(len(vertical_array), top_index + 6)
    top_window = vertical_array[start_idx:end_idx]

    top_steadiness = float(np.std(top_window)) if len(top_window) > 1 else 0

    valid_frames = task_signal_data["arm_raise_valid_frames"]
    signal_score = valid_frames / max(total_frames, 1)

    elbow_angle_mean = float(np.mean(elbow_angles)) if elbow_angles else None
    arm_distance_mean = float(np.mean(distance_array)) if len(
        distance_array) > 0 else None

    arm_raise_score = 100

    # Penalize shaky raise path.
    arm_raise_score -= min(30, arm_raise_smoothness * 80)

    # Penalize instability near the top.
    arm_raise_score -= min(25, top_steadiness * 100)

    # Penalize very small raise range.
    if raise_range < 0.12:
        arm_raise_score -= 20

    # Penalize bad signal.
    if signal_score < 0.75:
        arm_raise_score -= 15

    arm_raise_score = max(0, round(arm_raise_score, 1))

    if arm_raise_score >= 85:
        grade = "Excellent"
    elif arm_raise_score >= 70:
        grade = "Good"
    elif arm_raise_score >= 55:
        grade = "Moderate"
    else:
        grade = "Needs Work"

    return {
        "task_type": "arm_raise",
        "status": "arm_raise_v1_analysis",
        "summary": "Arm raise movement was analyzed using shoulder, elbow, and wrist motion.",
        "arm_raise_max": safe_float(raise_max),
        "arm_raise_min": safe_float(raise_min),
        "arm_raise_range": safe_float(raise_range),
        "arm_raise_smoothness": safe_float(arm_raise_smoothness),
        "top_steadiness": safe_float(top_steadiness),
        "arm_distance_mean": safe_float(arm_distance_mean),
        "elbow_angle_mean": safe_float(elbow_angle_mean),
        "arm_raise_valid_frames": valid_frames,
        "arm_raise_signal_score": round(signal_score, 2),
        "arm_raise_stability_score": arm_raise_score,
        "arm_raise_stability_grade": grade,
        "note": "This is an early arm raise analysis focused on raise range, smoothness, and top-position steadiness."
    }


def build_arm_raise_insights(task_analysis):

    insights = []

    if task_analysis is None:
        return ["Arm raise analysis unavailable."]

    if task_analysis.get("status") == "insufficient_arm_raise_data":
        return [
            "Not enough arm raise data was captured. Re-record with the shoulder, elbow, and wrist visible."
        ]

    raise_range = task_analysis.get("arm_raise_range")
    smoothness = task_analysis.get("arm_raise_smoothness")
    top_steadiness = task_analysis.get("top_steadiness")
    score = task_analysis.get("arm_raise_stability_score")
    signal_score = task_analysis.get("arm_raise_signal_score")

    if raise_range is not None:
        if raise_range < 0.12:
            insights.append(
                "Arm raise range was limited, suggesting the arm did not move very far upward or was not clearly captured."
            )
        else:
            insights.append(
                "Arm raise range was clearly detected."
            )

    if smoothness is not None:
        if smoothness > 0.025:
            insights.append(
                "Arm raise motion showed noticeable variability, suggesting reduced smoothness."
            )
        else:
            insights.append(
                "Arm raise motion was smooth and controlled."
            )

    if top_steadiness is not None:
        if top_steadiness > 0.025:
            insights.append(
                "The arm was less steady near the top of the raise."
            )
        else:
            insights.append(
                "The arm stayed relatively steady near the top of the raise."
            )

    if signal_score is not None:
        if signal_score < 0.75:
            insights.append(
                "Arm raise signal quality was limited, so the result should be interpreted cautiously."
            )
        else:
            insights.append(
                "Arm raise signal quality was usable for this analysis."
            )

    if score is not None:
        insights.append(
            f"Overall arm raise stability score: {score}/100."
        )

    return insights

# -----------------------------
# WALKING TASK ANALYSIS
# -----------------------------


def count_step_cycles(step_signal):
    if len(step_signal) < 5:
        return 0

    centered = np.array(step_signal) - np.mean(step_signal)
    signs = np.sign(centered)

    crossings = 0

    for i in range(1, len(signs)):
        if signs[i - 1] == 0 or signs[i] == 0:
            continue

        if signs[i - 1] != signs[i]:
            crossings += 1

    # A full left-right-left cycle usually creates about 2 crossings.
    return crossings // 2


def summarize_walking_task(task_signal_data, total_frames, fps: float = 30.0):
    hip_x = task_signal_data["walking_hip_center_x"]
    hip_y = task_signal_data["walking_hip_center_y"]
    step_signal = task_signal_data["walking_step_signal"]
    left_knee_angles = [
        v for v in task_signal_data["walking_left_knee_angles"] if v is not None
    ]
    right_knee_angles = [
        v for v in task_signal_data["walking_right_knee_angles"] if v is not None
    ]

    valid_frames = task_signal_data["walking_valid_frames"]
    signal_score = valid_frames / max(total_frames, 1)

    if signal_score >= 0.75:
        signal_grade = "Good"
    elif signal_score >= 0.45:
        signal_grade = "Moderate"
    else:
        signal_grade = "Poor"

    if len(step_signal) < 10 or len(hip_x) < 10:
        return {
            "task_type": "walking",
            "status": "insufficient_walking_data",
            "walking_valid_frames": valid_frames,
            "walking_signal_score": round(signal_score, 2),
            "walking_signal_grade": signal_grade,
            "estimated_step_cycles": 0,
            "walking_duration_sec": None,
            "cadence_estimate": None,
            "hip_path_range": None,
            "hip_vertical_variability": None,
            "step_rhythm_variability": None,
            "left_knee_range": None,
            "right_knee_range": None,
            "knee_range_difference": None,
            "walking_stability_score": None,
            "walking_stability_grade": "N/A",
            "summary": "Not enough walking data was captured.",
            "note": "Record a few clear walking steps with hips, knees, ankles, and feet visible."
        }

    step_array = np.array(step_signal)
    hip_x_array = np.array(hip_x)
    hip_y_array = np.array(hip_y)

    estimated_step_cycles = count_step_cycles(step_signal)
    duration_sec = len(step_signal) / fps if fps else len(step_signal) / 30

    cadence_estimate = None
    if duration_sec > 0:
        cadence_estimate = (estimated_step_cycles * 2 / duration_sec) * 60

    hip_path_range = float(np.max(hip_x_array) - np.min(hip_x_array))
    hip_vertical_variability = float(np.std(hip_y_array))
    step_rhythm_variability = float(
        np.std(np.diff(step_array))) if len(step_array) > 2 else None

    left_knee_range = None
    right_knee_range = None
    knee_range_difference = None

    if len(left_knee_angles) >= 5:
        left_knee_range = float(
            np.max(left_knee_angles) - np.min(left_knee_angles))

    if len(right_knee_angles) >= 5:
        right_knee_range = float(
            np.max(right_knee_angles) - np.min(right_knee_angles))

    if left_knee_range is not None and right_knee_range is not None:
        knee_range_difference = abs(left_knee_range - right_knee_range)

    walking_score = 100

    if signal_score < 0.75:
        walking_score -= 20
    elif signal_score < 0.9:
        walking_score -= 8

    if estimated_step_cycles < 2:
        walking_score -= 25

    if step_rhythm_variability is not None:
        walking_score -= min(25, step_rhythm_variability * 120)

    walking_score -= min(20, hip_vertical_variability * 150)

    if knee_range_difference is not None:
        walking_score -= min(20, knee_range_difference * 0.8)

    walking_score = max(0, round(walking_score, 1))

    if walking_score >= 85:
        walking_grade = "Excellent"
    elif walking_score >= 70:
        walking_grade = "Good"
    elif walking_score >= 55:
        walking_grade = "Moderate"
    else:
        walking_grade = "Needs Work"

    return {
        "task_type": "walking",
        "status": "walking_v1_analysis",
        "walking_valid_frames": valid_frames,
        "walking_signal_score": round(signal_score, 2),
        "walking_signal_grade": signal_grade,
        "estimated_step_cycles": estimated_step_cycles,
        "walking_duration_sec": safe_float(duration_sec),
        "cadence_estimate": safe_float(cadence_estimate),
        "hip_path_range": safe_float(hip_path_range),
        "hip_vertical_variability": safe_float(hip_vertical_variability),
        "step_rhythm_variability": safe_float(step_rhythm_variability),
        "left_knee_range": safe_float(left_knee_range),
        "right_knee_range": safe_float(right_knee_range),
        "knee_range_difference": safe_float(knee_range_difference),
        "walking_stability_score": walking_score,
        "walking_stability_grade": walking_grade,
        "summary": "Walking was analyzed using lower-body MediaPipe landmarks.",
        "note": "Walking V1 estimates rhythm and stability from hips, knees, and ankles. It is not a medical gait diagnosis."
    }


def build_walking_insights(task_analysis):
    insights = []

    if task_analysis is None:
        return ["Walking analysis was not available for this recording."]

    if task_analysis.get("status") == "insufficient_walking_data":
        return [
            "Not enough walking data was captured. Re-record with the full lower body visible for several steps."
        ]

    step_cycles = task_analysis.get("estimated_step_cycles")
    cadence = task_analysis.get("cadence_estimate")
    hip_vertical_variability = task_analysis.get("hip_vertical_variability")
    rhythm_variability = task_analysis.get("step_rhythm_variability")
    knee_difference = task_analysis.get("knee_range_difference")
    score = task_analysis.get("walking_stability_score")
    signal_grade = task_analysis.get("walking_signal_grade")

    if step_cycles is not None:
        if step_cycles < 2:
            insights.append(
                "Only a small number of walking cycles were detected, so this result is less reliable."
            )
        else:
            insights.append(
                f"Estimated walking cycles detected: {step_cycles}."
            )

    if cadence is not None:
        insights.append(
            f"Estimated cadence: {round(cadence, 1)} steps per minute."
        )

    if rhythm_variability is not None:
        if rhythm_variability > 0.04:
            insights.append(
                "Step rhythm showed noticeable variability during the recording."
            )
        else:
            insights.append(
                "Step rhythm appeared relatively consistent during the recording."
            )

    if hip_vertical_variability is not None:
        if hip_vertical_variability > 0.04:
            insights.append(
                "Hip height varied noticeably, which may suggest bouncing motion, camera angle issues, or unstable walking."
            )
        else:
            insights.append(
                "Hip height stayed relatively stable during walking."
            )

    if knee_difference is not None:
        if knee_difference > 15:
            insights.append(
                "Left and right knee motion ranges differed noticeably."
            )
        else:
            insights.append(
                "Left and right knee motion ranges were relatively similar."
            )

    if signal_grade == "Poor":
        insights.append(
            "Walking signal quality was poor, so treat this result as low confidence."
        )
    elif signal_grade == "Moderate":
        insights.append(
            "Walking signal quality was moderate, so this result is usable but should be interpreted carefully."
        )
    elif signal_grade == "Good":
        insights.append(
            "Walking signal quality was good for this analysis."
        )

    if score is not None:
        insights.append(
            f"Overall walking stability score: {score}/100."
        )

    return insights


# -----------------------------
# BALANCE TASK ANALYSIS
# -----------------------------
def summarize_balance_task(task_signal_data, total_frames):
    hip_x = task_signal_data["balance_hip_center_x"]
    hip_y = task_signal_data["balance_hip_center_y"]
    shoulder_x = task_signal_data["balance_shoulder_center_x"]
    shoulder_y = task_signal_data["balance_shoulder_center_y"]
    torso_lean_values = [
        v for v in task_signal_data["balance_torso_lean"] if v is not None
    ]
    ankle_center_x = task_signal_data["balance_ankle_center_x"]
    support_width_values = task_signal_data["balance_support_width"]

    valid_frames = task_signal_data["balance_valid_frames"]
    signal_score = valid_frames / max(total_frames, 1)

    if signal_score >= 0.75:
        signal_grade = "Good"
    elif signal_score >= 0.45:
        signal_grade = "Moderate"
    else:
        signal_grade = "Poor"

    if len(hip_x) < 10 or len(shoulder_x) < 10 or len(torso_lean_values) < 10:
        return {
            "task_type": "balance",
            "status": "insufficient_balance_data",
            "balance_valid_frames": valid_frames,
            "balance_signal_score": round(signal_score, 2),
            "balance_signal_grade": signal_grade,
            "hip_sway_range": None,
            "hip_vertical_variability": None,
            "shoulder_sway_range": None,
            "shoulder_vertical_variability": None,
            "torso_lean_mean": None,
            "torso_lean_max": None,
            "torso_lean_variability": None,
            "ankle_center_drift": None,
            "support_width_mean": None,
            "balance_stability_score": None,
            "balance_stability_grade": "N/A",
            "summary": "Not enough balance data was captured.",
            "note": "Re-record while standing still with your full body visible."
        }

    hip_x_array = np.array(hip_x)
    hip_y_array = np.array(hip_y)
    shoulder_x_array = np.array(shoulder_x)
    shoulder_y_array = np.array(shoulder_y)
    torso_array = np.array(torso_lean_values)
    ankle_center_array = np.array(ankle_center_x)
    support_width_array = np.array(support_width_values)

    hip_sway_range = float(np.max(hip_x_array) - np.min(hip_x_array))
    hip_vertical_variability = float(np.std(hip_y_array))

    shoulder_sway_range = float(
        np.max(shoulder_x_array) - np.min(shoulder_x_array))
    shoulder_vertical_variability = float(np.std(shoulder_y_array))

    torso_lean_mean = float(np.mean(torso_array))
    torso_lean_max = float(np.max(torso_array))
    torso_lean_variability = float(np.std(torso_array))

    ankle_center_drift = float(
        np.max(ankle_center_array) - np.min(ankle_center_array))
    support_width_mean = float(np.mean(support_width_array)) if len(
        support_width_array) > 0 else None

    balance_score = 100

    if signal_score < 0.75:
        balance_score -= 20
    elif signal_score < 0.9:
        balance_score -= 8

    balance_score -= min(25, hip_sway_range * 180)
    balance_score -= min(20, shoulder_sway_range * 150)
    balance_score -= min(20, torso_lean_variability * 2.5)
    balance_score -= min(15, hip_vertical_variability * 180)
    balance_score -= min(10, ankle_center_drift * 120)

    if torso_lean_max > 25:
        balance_score -= 10

    balance_score = max(0, round(balance_score, 1))

    if balance_score >= 85:
        balance_grade = "Excellent"
    elif balance_score >= 70:
        balance_grade = "Good"
    elif balance_score >= 55:
        balance_grade = "Moderate"
    else:
        balance_grade = "Needs Work"

    return {
        "task_type": "balance",
        "status": "balance_v1_analysis",
        "balance_valid_frames": valid_frames,
        "balance_signal_score": round(signal_score, 2),
        "balance_signal_grade": signal_grade,
        "hip_sway_range": safe_float(hip_sway_range),
        "hip_vertical_variability": safe_float(hip_vertical_variability),
        "shoulder_sway_range": safe_float(shoulder_sway_range),
        "shoulder_vertical_variability": safe_float(shoulder_vertical_variability),
        "torso_lean_mean": safe_float(torso_lean_mean),
        "torso_lean_max": safe_float(torso_lean_max),
        "torso_lean_variability": safe_float(torso_lean_variability),
        "ankle_center_drift": safe_float(ankle_center_drift),
        "support_width_mean": safe_float(support_width_mean),
        "balance_stability_score": balance_score,
        "balance_stability_grade": balance_grade,
        "summary": "Standing balance was analyzed using body-center sway and torso stability.",
        "note": "Balance V1 estimates visible postural stability. It is not a fall-risk diagnosis."
    }


def build_balance_insights(task_analysis):
    insights = []

    if task_analysis is None:
        return ["Balance analysis was not available for this recording."]

    if task_analysis.get("status") == "insufficient_balance_data":
        return [
            "Not enough balance data was captured. Re-record while standing still with your full body visible."
        ]

    hip_sway = task_analysis.get("hip_sway_range")
    shoulder_sway = task_analysis.get("shoulder_sway_range")
    torso_variability = task_analysis.get("torso_lean_variability")
    torso_max = task_analysis.get("torso_lean_max")
    ankle_drift = task_analysis.get("ankle_center_drift")
    signal_grade = task_analysis.get("balance_signal_grade")
    score = task_analysis.get("balance_stability_score")

    if hip_sway is not None:
        if hip_sway > 0.06:
            insights.append(
                "Hip center shifted noticeably during the balance check."
            )
        else:
            insights.append(
                "Hip center stayed relatively steady during the balance check."
            )

    if shoulder_sway is not None:
        if shoulder_sway > 0.06:
            insights.append(
                "Shoulder position drifted noticeably, suggesting visible upper-body sway."
            )
        else:
            insights.append(
                "Shoulder position stayed relatively steady."
            )

    if torso_variability is not None:
        if torso_variability > 5:
            insights.append(
                "Torso lean varied noticeably during the check."
            )
        else:
            insights.append(
                "Torso lean stayed relatively consistent."
            )

    if torso_max is not None:
        if torso_max > 25:
            insights.append(
                "Maximum torso lean was high, so posture may not have stayed centered."
            )
        else:
            insights.append(
                "Maximum torso lean stayed within a controlled range."
            )

    if ankle_drift is not None:
        if ankle_drift > 0.04:
            insights.append(
                "Foot or ankle center appeared to drift, which may reflect stepping, shifting, or camera tracking noise."
            )
        else:
            insights.append(
                "Foot position appeared relatively stable."
            )

    if signal_grade == "Poor":
        insights.append(
            "Balance signal quality was poor, so treat this result as low confidence."
        )
    elif signal_grade == "Moderate":
        insights.append(
            "Balance signal quality was moderate, so the result is usable but should be interpreted carefully."
        )
    elif signal_grade == "Good":
        insights.append(
            "Balance signal quality was good for this analysis."
        )

    if score is not None:
        insights.append(
            f"Overall balance stability score: {score}/100."
        )

    return insights

# -----------------------------
# TIMED UP AND GO TASK ANALYSIS
# -----------------------------


def count_direction_changes(signal, min_change=0.02):
    if len(signal) < 8:
        return 0

    arr = np.array(signal)
    smoothed = moving_average(arr.tolist(), window_size=5)
    diffs = []

    for i in range(1, len(smoothed)):
        if smoothed[i] is None or smoothed[i - 1] is None:
            continue
        diffs.append(smoothed[i] - smoothed[i - 1])

    if len(diffs) < 5:
        return 0

    signs = []

    for diff in diffs:
        if abs(diff) < min_change:
            signs.append(0)
        elif diff > 0:
            signs.append(1)
        else:
            signs.append(-1)

    cleaned = [s for s in signs if s != 0]

    if len(cleaned) < 2:
        return 0

    changes = 0

    for i in range(1, len(cleaned)):
        if cleaned[i] != cleaned[i - 1]:
            changes += 1

    return changes


def summarize_tug_task(task_signal_data, total_frames, fps: float = 30.0):
    hip_x = task_signal_data["tug_hip_center_x"]
    hip_y = task_signal_data["tug_hip_center_y"]
    knee_angles = [
        v for v in task_signal_data["tug_right_knee_angles"] if v is not None
    ]
    hip_angles = [
        v for v in task_signal_data["tug_right_hip_angles"] if v is not None
    ]
    torso_lean_values = [
        v for v in task_signal_data["tug_torso_lean"] if v is not None
    ]
    step_signal = task_signal_data["tug_step_signal"]

    valid_frames = task_signal_data["tug_valid_frames"]
    signal_score = valid_frames / max(total_frames, 1)

    if signal_score >= 0.75:
        signal_grade = "Good"
    elif signal_score >= 0.45:
        signal_grade = "Moderate"
    else:
        signal_grade = "Poor"

    if len(hip_x) < 15 or len(knee_angles) < 10 or len(step_signal) < 10:
        return {
            "task_type": "timed_up_and_go",
            "status": "insufficient_tug_data",
            "tug_valid_frames": valid_frames,
            "tug_signal_score": round(signal_score, 2),
            "tug_signal_grade": signal_grade,
            "tug_duration_sec": None,
            "tug_path_range": None,
            "tug_return_pattern_detected": False,
            "tug_direction_changes": 0,
            "tug_estimated_step_cycles": 0,
            "tug_cadence_estimate": None,
            "tug_knee_range": None,
            "tug_hip_range": None,
            "tug_torso_lean_max": None,
            "tug_torso_lean_variability": None,
            "tug_hip_vertical_variability": None,
            "tug_mobility_score": None,
            "tug_mobility_grade": "N/A",
            "summary": "Not enough Timed Up and Go data was captured.",
            "note": "Record the full sequence: sit, stand, walk, turn or return, and finish clearly in frame."
        }

    hip_x_array = np.array(hip_x)
    hip_y_array = np.array(hip_y)
    knee_array = np.array(knee_angles)
    hip_angle_array = np.array(hip_angles) if hip_angles else np.array([])
    torso_array = np.array(
        torso_lean_values) if torso_lean_values else np.array([])

    tug_duration_sec = len(hip_x) / fps if fps else len(hip_x) / 30.0

    tug_path_range = float(np.max(hip_x_array) - np.min(hip_x_array))
    tug_hip_vertical_variability = float(np.std(hip_y_array))

    tug_direction_changes = count_direction_changes(hip_x)
    tug_return_pattern_detected = tug_direction_changes >= 1 and tug_path_range > 0.08

    tug_estimated_step_cycles = count_step_cycles(step_signal)

    tug_cadence_estimate = None
    if tug_duration_sec > 0:
        tug_cadence_estimate = (
            tug_estimated_step_cycles * 2 / tug_duration_sec) * 60

    tug_knee_range = float(np.max(knee_array) - np.min(knee_array))

    tug_hip_range = None
    if len(hip_angle_array) >= 5:
        tug_hip_range = float(np.max(hip_angle_array) -
                              np.min(hip_angle_array))

    tug_torso_lean_max = None
    tug_torso_lean_variability = None

    if len(torso_array) >= 5:
        tug_torso_lean_max = float(np.max(torso_array))
        tug_torso_lean_variability = float(np.std(torso_array))

    tug_score = 100

    if signal_score < 0.75:
        tug_score -= 20
    elif signal_score < 0.9:
        tug_score -= 8

    if tug_duration_sec < 3:
        tug_score -= 15

    if tug_path_range < 0.08:
        tug_score -= 20

    if not tug_return_pattern_detected:
        tug_score -= 15

    if tug_estimated_step_cycles < 2:
        tug_score -= 15

    if tug_knee_range < 15:
        tug_score -= 10

    if tug_hip_vertical_variability > 0.05:
        tug_score -= min(15, tug_hip_vertical_variability * 150)

    if tug_torso_lean_max is not None and tug_torso_lean_max > 35:
        tug_score -= 10

    if tug_torso_lean_variability is not None:
        tug_score -= min(15, tug_torso_lean_variability * 1.5)

    tug_score = max(0, round(tug_score, 1))

    if tug_score >= 85:
        tug_grade = "Excellent"
    elif tug_score >= 70:
        tug_grade = "Good"
    elif tug_score >= 55:
        tug_grade = "Moderate"
    else:
        tug_grade = "Needs Work"

    return {
        "task_type": "timed_up_and_go",
        "status": "tug_v1_analysis",
        "tug_valid_frames": valid_frames,
        "tug_signal_score": round(signal_score, 2),
        "tug_signal_grade": signal_grade,
        "tug_duration_sec": safe_float(tug_duration_sec),
        "tug_path_range": safe_float(tug_path_range),
        "tug_return_pattern_detected": tug_return_pattern_detected,
        "tug_direction_changes": tug_direction_changes,
        "tug_estimated_step_cycles": tug_estimated_step_cycles,
        "tug_cadence_estimate": safe_float(tug_cadence_estimate),
        "tug_knee_range": safe_float(tug_knee_range),
        "tug_hip_range": safe_float(tug_hip_range),
        "tug_torso_lean_max": safe_float(tug_torso_lean_max),
        "tug_torso_lean_variability": safe_float(tug_torso_lean_variability),
        "tug_hip_vertical_variability": safe_float(tug_hip_vertical_variability),
        "tug_mobility_score": tug_score,
        "tug_mobility_grade": tug_grade,
        "summary": "Timed Up and Go was analyzed as a functional mobility sequence.",
        "note": "TUG V1 estimates full-sequence mobility from body path, stepping rhythm, knee motion, torso control, and return-pattern detection."
    }


def build_tug_insights(task_analysis):
    insights = []

    if task_analysis is None:
        return ["Timed Up and Go analysis was not available for this recording."]

    if task_analysis.get("status") == "insufficient_tug_data":
        return [
            "Not enough Timed Up and Go data was captured. Re-record the full sit, stand, walk, turn or return sequence."
        ]

    duration = task_analysis.get("tug_duration_sec")
    path_range = task_analysis.get("tug_path_range")
    return_detected = task_analysis.get("tug_return_pattern_detected")
    step_cycles = task_analysis.get("tug_estimated_step_cycles")
    cadence = task_analysis.get("tug_cadence_estimate")
    knee_range = task_analysis.get("tug_knee_range")
    torso_max = task_analysis.get("tug_torso_lean_max")
    torso_variability = task_analysis.get("tug_torso_lean_variability")
    signal_grade = task_analysis.get("tug_signal_grade")
    score = task_analysis.get("tug_mobility_score")

    if duration is not None:
        insights.append(
            f"Estimated visible sequence duration: {round(duration, 1)} seconds."
        )

    if path_range is not None:
        if path_range < 0.08:
            insights.append(
                "Body path movement was small, so the walking/return part may not have been fully captured."
            )
        else:
            insights.append(
                "Body path movement was clearly detected during the sequence."
            )

    if return_detected:
        insights.append(
            "A return or direction-change pattern was detected."
        )
    else:
        insights.append(
            "A clear return or direction-change pattern was not detected."
        )

    if step_cycles is not None:
        if step_cycles < 2:
            insights.append(
                "Only a small number of walking cycles were detected."
            )
        else:
            insights.append(
                f"Estimated walking cycles during the sequence: {step_cycles}."
            )

    if cadence is not None:
        insights.append(
            f"Estimated cadence during the sequence: {round(cadence, 1)} steps per minute."
        )

    if knee_range is not None:
        if knee_range < 15:
            insights.append(
                "Knee motion range was limited, which may mean the sit-to-stand or walking phase was incomplete or poorly captured."
            )
        else:
            insights.append(
                "Knee motion changed clearly during the sequence."
            )

    if torso_max is not None:
        if torso_max > 35:
            insights.append(
                "Torso lean was high during the sequence."
            )
        else:
            insights.append(
                "Torso lean stayed within a controlled range."
            )

    if torso_variability is not None:
        if torso_variability > 6:
            insights.append(
                "Torso motion varied noticeably during the sequence."
            )
        else:
            insights.append(
                "Torso motion stayed relatively consistent."
            )

    if signal_grade == "Poor":
        insights.append(
            "TUG signal quality was poor, so treat this result as low confidence."
        )
    elif signal_grade == "Moderate":
        insights.append(
            "TUG signal quality was moderate, so this result is usable but should be interpreted carefully."
        )
    elif signal_grade == "Good":
        insights.append(
            "TUG signal quality was good for this analysis."
        )

    if score is not None:
        insights.append(
            f"Overall functional mobility score: {score}/100."
        )

    return insights


def safe_float(x):
    return float(x) if x is not None else None


# -----------------------------
# VIDEO PROCESSING
# -----------------------------
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps is None or fps == 0:
    fps = 30.0

fps = float(fps)

if not cap.isOpened():
    error_result = {
        "error": f"Could not open video: {video_path}"
    }
    if BACKEND_MODE:
        print(json.dumps(error_result))
    else:
        print(f"Error: Could not open video: {video_path}")
    raise SystemExit(1)

frame_index = 0
angle_data = []

# -----------------------------
# TASK-SPECIFIC SIGNAL STORAGE
# -----------------------------
task_signal_data = {
    # Sit-to-Stand
    "right_knee_angles": [],
    "right_hip_angles": [],
    "torso_lean_values": [],
    "sit_to_stand_valid_frames": 0,

    # Reach
    "reach_distances": [],
    "reach_x_positions": [],
    "reach_y_positions": [],
    "reach_valid_frames": 0,

    # Arm Raise
    "arm_raise_vertical_positions": [],
    "arm_raise_distances": [],
    "arm_raise_elbow_angles": [],
    "arm_raise_valid_frames": 0,

    # Walking
    "walking_hip_center_x": [],
    "walking_hip_center_y": [],
    "walking_left_ankle_y": [],
    "walking_right_ankle_y": [],
    "walking_left_knee_angles": [],
    "walking_right_knee_angles": [],
    "walking_step_signal": [],
    "walking_valid_frames": 0,

    # Balance
    "balance_hip_center_x": [],
    "balance_hip_center_y": [],
    "balance_shoulder_center_x": [],
    "balance_shoulder_center_y": [],
    "balance_torso_lean": [],
    "balance_ankle_center_x": [],
    "balance_support_width": [],
    "balance_valid_frames": 0,

    # Timed Up and Go
    "tug_hip_center_x": [],
    "tug_hip_center_y": [],
    "tug_right_knee_angles": [],
    "tug_right_hip_angles": [],
    "tug_torso_lean": [],
    "tug_step_signal": [],
    "tug_valid_frames": 0
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    angle = None

    pose_landmarks = getattr(results, "pose_landmarks", None)

    if pose_landmarks is not None:
        if not BACKEND_MODE:
            mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                getattr(mp_pose, "POSE_CONNECTIONS")
            )

        landmarks = pose_landmarks.landmark

        shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ]
        elbow = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
        ]
        wrist = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        ]

        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]

        # -----------------------------
        # SIT-TO-STAND TASK LANDMARKS
        # -----------------------------
        right_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        ]

        right_knee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        ]

        right_ankle = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        ]

        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        ]

        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        ]

        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        ]

        angle = calculate_angle(shoulder, elbow, wrist)

        # -----------------------------
        # SIT-TO-STAND TASK SIGNALS
        # -----------------------------
        if MODE == "daily" and DAILY_TASK == "sit_to_stand":
            right_knee_angle = calculate_angle(
                right_hip, right_knee, right_ankle)
            right_hip_angle = calculate_angle(shoulder, right_hip, right_knee)
            torso_lean = calculate_torso_lean(shoulder, right_hip)

            task_signal_data["right_knee_angles"].append(right_knee_angle)
            task_signal_data["right_hip_angles"].append(right_hip_angle)
            task_signal_data["torso_lean_values"].append(torso_lean)

            if right_knee_angle is not None and right_hip_angle is not None and torso_lean is not None:
                task_signal_data["sit_to_stand_valid_frames"] += 1

        # -----------------------------
        # REACH TASK SIGNALS
        # -----------------------------
        if MODE == "daily" and DAILY_TASK == "reach":

            reach_distance = np.linalg.norm(
                np.array(wrist) - np.array(shoulder)
            )

            task_signal_data["reach_distances"].append(float(reach_distance))
            task_signal_data["reach_x_positions"].append(float(wrist[0]))
            task_signal_data["reach_y_positions"].append(float(wrist[1]))

            task_signal_data["reach_valid_frames"] += 1

        if angle is not None and not BACKEND_MODE:
            h, w, _ = frame.shape
            cx = int(elbow[0] * w)
            cy = int(elbow[1] * h)

            cv2.putText(
                frame,
                f"{int(angle)} deg",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # -----------------------------
        # ARM RAISE TASK SIGNALS
        # -----------------------------
        if MODE == "daily" and DAILY_TASK == "arm_raise":

            # In image coordinates, y gets smaller when the wrist moves upward.
            # shoulder_y - wrist_y becomes larger when the arm is raised.
            arm_vertical_position = shoulder[1] - wrist[1]

            arm_distance = np.linalg.norm(
                np.array(wrist) - np.array(shoulder)
            )

            task_signal_data["arm_raise_vertical_positions"].append(
                float(arm_vertical_position)
            )
            task_signal_data["arm_raise_distances"].append(float(arm_distance))
            task_signal_data["arm_raise_elbow_angles"].append(angle)

            if angle is not None:
                task_signal_data["arm_raise_valid_frames"] += 1

        # -----------------------------
        # WALKING TASK SIGNALS
        # -----------------------------
        if MODE == "daily" and DAILY_TASK == "walking":
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2

            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(
                right_hip, right_knee, right_ankle)

            # This simple signal compares left/right ankle vertical motion.
            # During walking, the two ankles should alternate.
            step_signal = left_ankle[1] - right_ankle[1]

            task_signal_data["walking_hip_center_x"].append(
                float(hip_center_x))
            task_signal_data["walking_hip_center_y"].append(
                float(hip_center_y))
            task_signal_data["walking_left_ankle_y"].append(
                float(left_ankle[1]))
            task_signal_data["walking_right_ankle_y"].append(
                float(right_ankle[1]))
            task_signal_data["walking_left_knee_angles"].append(
                left_knee_angle)
            task_signal_data["walking_right_knee_angles"].append(
                right_knee_angle)
            task_signal_data["walking_step_signal"].append(float(step_signal))

            if left_knee_angle is not None and right_knee_angle is not None:
                task_signal_data["walking_valid_frames"] += 1

        # -----------------------------
        # BALANCE TASK SIGNALS
        # -----------------------------
        if MODE == "daily" and DAILY_TASK == "balance":
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2

            shoulder_center_x = (left_shoulder[0] + shoulder[0]) / 2
            shoulder_center_y = (left_shoulder[1] + shoulder[1]) / 2

            ankle_center_x = (left_ankle[0] + right_ankle[0]) / 2
            support_width = abs(left_ankle[0] - right_ankle[0])

            torso_lean = calculate_torso_lean(
                [shoulder_center_x, shoulder_center_y],
                [hip_center_x, hip_center_y]
            )

            task_signal_data["balance_hip_center_x"].append(
                float(hip_center_x))
            task_signal_data["balance_hip_center_y"].append(
                float(hip_center_y))
            task_signal_data["balance_shoulder_center_x"].append(
                float(shoulder_center_x))
            task_signal_data["balance_shoulder_center_y"].append(
                float(shoulder_center_y))
            task_signal_data["balance_ankle_center_x"].append(
                float(ankle_center_x))
            task_signal_data["balance_support_width"].append(
                float(support_width))
            task_signal_data["balance_torso_lean"].append(torso_lean)

            if torso_lean is not None:
                task_signal_data["balance_valid_frames"] += 1

        # -----------------------------
        # TIMED UP AND GO TASK SIGNALS
        # -----------------------------
        if MODE == "daily" and DAILY_TASK == "timed_up_and_go":
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2

            right_knee_angle = calculate_angle(
                right_hip, right_knee, right_ankle
            )
            right_hip_angle = calculate_angle(
                shoulder, right_hip, right_knee
            )

            torso_lean = calculate_torso_lean(shoulder, right_hip)

            # Left/right ankle vertical difference gives a rough stepping signal.
            step_signal = left_ankle[1] - right_ankle[1]

            task_signal_data["tug_hip_center_x"].append(float(hip_center_x))
            task_signal_data["tug_hip_center_y"].append(float(hip_center_y))
            task_signal_data["tug_right_knee_angles"].append(right_knee_angle)
            task_signal_data["tug_right_hip_angles"].append(right_hip_angle)
            task_signal_data["tug_torso_lean"].append(torso_lean)
            task_signal_data["tug_step_signal"].append(float(step_signal))

            if (
                right_knee_angle is not None
                and right_hip_angle is not None
                and torso_lean is not None
            ):
                task_signal_data["tug_valid_frames"] += 1

    angle_data.append([frame_index, angle])

    if not BACKEND_MODE:
        cv2.imshow("Biomechanics Analysis", frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    frame_index += 1

cap.release()

if not BACKEND_MODE:
    cv2.destroyAllWindows()


# -----------------------------
# DATA PROCESSING
# -----------------------------
angles = [row[1] for row in angle_data]
signal_quality = compute_signal_quality(angles)
smoothed_angles = moving_average(angles, window_size=5)


# -----------------------------
# ANGULAR VELOCITY
# -----------------------------
angular_velocity = []

for i in range(len(smoothed_angles)):
    if i == 0 or smoothed_angles[i] is None or smoothed_angles[i - 1] is None:
        angular_velocity.append(None)
    else:
        angular_velocity.append(smoothed_angles[i] - smoothed_angles[i - 1])


# Velocity metrics
valid_velocities = [v for v in angular_velocity if v is not None]
max_velocity = max(valid_velocities) if valid_velocities else None
min_velocity = min(valid_velocities) if valid_velocities else None

if not BACKEND_MODE and valid_velocities:
    print(f"Max extension speed: {max_velocity:.2f} deg/frame")
    print(f"Max flexion speed: {min_velocity:.2f} deg/frame")


# -----------------------------
# REP COUNTING
# -----------------------------
transitions, reps = count_reps_bidirectional(smoothed_angles)

if not BACKEND_MODE:
    print("\n--- RESULTS ---")
    print(f"Total frames: {len(angle_data)}")
    print(f"State transitions: {transitions}")
    print(f"Estimated reps: {reps}")


# -----------------------------
# QUALITY METRICS (GLOBAL)
# -----------------------------
global_smoothness = compute_smoothness(angular_velocity)
global_symmetry = compute_symmetry(angular_velocity)
global_control = compute_control(angular_velocity)

if not BACKEND_MODE:
    print("\n--- QUALITY METRICS ---")
    print(f"Smoothness: {global_smoothness:.3f}")
    print(f"Symmetry: {global_symmetry:.3f}")
    print(f"Control: {global_control:.3f}")


# -----------------------------
# PER-REP ANALYSIS
# -----------------------------
transition_indices = get_transition_indices(smoothed_angles)
rep_segments = segment_reps(smoothed_angles, transition_indices)
rep_results = analyze_reps(rep_segments, angular_velocity)
fatigue_analysis = analyze_fatigue(rep_results)

if not BACKEND_MODE:
    print("\n--- PER-REP ANALYSIS ---")

    for r in rep_results:
        print(f"\nRep {r['rep']} (frames {r['start']} → {r['end']}):")

        smooth_val = r["smoothness"]
        sym_val = r["symmetry"]
        ctrl_val = r["control"]

        if smooth_val is not None:
            print(
                f"  Smoothness: {grade_smoothness(smooth_val)} ({smooth_val:.3f})")
        else:
            print("  Smoothness: N/A")

        if sym_val is not None:
            print(f"  Symmetry: {grade_symmetry(sym_val)} ({sym_val:.3f})")
        else:
            print("  Symmetry: N/A")

        if ctrl_val is not None:
            print(f"  Control: {grade_control(ctrl_val)} ({ctrl_val:.3f})")
        else:
            print("  Control: N/A")

    print("\n--- FATIGUE ANALYSIS ---")
    print(fatigue_analysis)


# -----------------------------
# SAVE CSV
# -----------------------------
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["frame", "raw_angle", "smoothed_angle", "angular_velocity"])

    for i in range(len(angle_data)):
        writer.writerow([
            i,
            angles[i],
            smoothed_angles[i],
            angular_velocity[i]
        ])

if not BACKEND_MODE:
    print(f"\nSaved data to {output_csv}")


# -----------------------------
# PLOTS
# -----------------------------
plot_angles = [a if a is not None else np.nan for a in smoothed_angles]
plot_velocity = [v if v is not None else np.nan for v in angular_velocity]

if not BACKEND_MODE:
    plt.figure(figsize=(12, 6))
    plt.plot(plot_angles)
    plt.axhline(y=70, linestyle="--")
    plt.axhline(y=140, linestyle="--")
    plt.title("Angle")
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(plot_velocity)
    plt.axhline(y=0)
    plt.title("Velocity")
    plt.grid()
    plt.show()


# -----------------------------
# FINAL JSON OUTPUT FOR SERVER
# -----------------------------
global_metrics_payload = {
    "smoothness": safe_float(global_smoothness),
    "symmetry": safe_float(global_symmetry),
    "control": safe_float(global_control),
    "efficiency": safe_float(compute_efficiency(
        global_smoothness,
        global_symmetry,
        global_control
    )),
    "smoothness_grade": grade_smoothness(global_smoothness),
    "symmetry_grade": grade_symmetry(global_symmetry),
    "control_grade": grade_control(global_control),
    "efficiency_grade": grade_efficiency(
        compute_efficiency(
            global_smoothness,
            global_symmetry,
            global_control
        )
    ),
}

result_data = {
    "mode": MODE,
    "daily_task": DAILY_TASK,
    "daily_task_label": get_daily_task_label(DAILY_TASK),
    "daily_task_focus": get_daily_task_focus(DAILY_TASK),
    "video_path": video_path,
    "output_csv": output_csv,
    "total_frames": len(angle_data),
    "signal_quality": signal_quality,
    "reps": safe_float(reps),
    "transitions": transitions,
    "max_extension_speed": safe_float(max_velocity),
    "max_flexion_speed": safe_float(min_velocity),
    "fatigue_analysis": fatigue_analysis,
    "global_metrics": global_metrics_payload,
    "interpretation": interpret_results(MODE, global_metrics_payload, fatigue_analysis, DAILY_TASK),
    "rep_analysis": []
}

for r in rep_results:
    smooth_val = r["smoothness"]
    sym_val = r["symmetry"]
    ctrl_val = r["control"]

    score = compute_rep_score(
        r["smoothness"],
        r["symmetry"],
        r["control"]
    )

    result_data["rep_analysis"].append({
        "rep": r["rep"],
        "start": r["start"],
        "end": r["end"],
        "smoothness": safe_float(smooth_val),
        "symmetry": safe_float(sym_val),
        "control": safe_float(ctrl_val),
        "smoothness_grade": grade_smoothness(smooth_val),
        "symmetry_grade": grade_symmetry(sym_val),
        "control_grade": grade_control(ctrl_val),
        "score": score,
        "score_grade": grade_score(score)
    })

    # -----------------------------
# DEGRADATION DATA
# -----------------------------
rep_scores = [
    r["score"] for r in result_data["rep_analysis"]
    if r["score"] is not None
]

result_data["score_trend"] = rep_scores

if len(rep_scores) >= 2:
    result_data["performance_drop"] = rep_scores[0] - rep_scores[-1]
else:
    result_data["performance_drop"] = None


result_data["performance_summary"] = interpret_degradation(
    result_data["performance_drop"]
)

movement_health_score = compute_movement_health_score(
    result_data["global_metrics"],
    result_data["performance_drop"]
)

result_data["movement_health_score"] = movement_health_score
result_data["movement_health_grade"] = grade_movement_health(
    movement_health_score
)

result_data["key_insights"] = build_key_insights(
    result_data["global_metrics"],
    result_data["rep_analysis"],
    result_data["performance_summary"],
    MODE
)

result_data["movement_signature"] = classify_movement_signature(
    result_data["global_metrics"],
    result_data["performance_summary"],
    MODE
)

# -----------------------------
# TASK-SPECIFIC ANALYSIS OUTPUT
# -----------------------------
result_data["task_analysis"] = None

if MODE == "daily" and DAILY_TASK == "sit_to_stand":
    task_analysis: Dict[str, Any] = summarize_sit_to_stand_task(
        task_signal_data,
        len(angle_data),
        fps
    )

    task_analysis["task_insights"] = build_sit_to_stand_insights(
        task_analysis
    )

    result_data["task_analysis"] = task_analysis

elif MODE == "daily" and DAILY_TASK == "reach":
    task_analysis: Dict[str, Any] = summarize_reach_task(
        task_signal_data,
        len(angle_data)
    )

    task_analysis["task_insights"] = build_reach_insights(
        task_analysis
    )

    result_data["task_analysis"] = task_analysis

elif MODE == "daily" and DAILY_TASK == "arm_raise":
    task_analysis: Dict[str, Any] = summarize_arm_raise_task(
        task_signal_data,
        len(angle_data)
    )

    task_analysis["task_insights"] = build_arm_raise_insights(
        task_analysis
    )

    result_data["task_analysis"] = task_analysis

elif MODE == "daily" and DAILY_TASK == "walking":
    task_analysis: Dict[str, Any] = summarize_walking_task(
        task_signal_data,
        len(angle_data),
        fps
    )

    task_analysis["task_insights"] = build_walking_insights(
        task_analysis
    )

    result_data["task_analysis"] = task_analysis

elif MODE == "daily" and DAILY_TASK == "balance":
    task_analysis: Dict[str, Any] = summarize_balance_task(
        task_signal_data,
        len(angle_data)
    )

    task_analysis["task_insights"] = build_balance_insights(
        task_analysis
    )

    result_data["task_analysis"] = task_analysis

elif MODE == "daily" and DAILY_TASK == "timed_up_and_go":
    task_analysis: Dict[str, Any] = summarize_tug_task(
        task_signal_data,
        len(angle_data),
        fps
    )

    task_analysis["task_insights"] = build_tug_insights(
        task_analysis
    )

    result_data["task_analysis"] = task_analysis

# -----------------------------
# BEST / WORST REP DETECTION
# -----------------------------
valid_reps = [
    r for r in result_data["rep_analysis"]
    if r["score"] is not None
]

if valid_reps:
    best_rep = max(valid_reps, key=lambda x: x["score"])
    worst_rep = min(valid_reps, key=lambda x: x["score"])

    result_data["best_rep"] = best_rep
    result_data["worst_rep"] = worst_rep
else:
    result_data["best_rep"] = None
    result_data["worst_rep"] = None

if BACKEND_MODE:
    print(json.dumps(result_data))
