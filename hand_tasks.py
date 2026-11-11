"""
hand_tasks.py -- Phase 1 hand-based movement checks: Finger Tapping
(Camera), Hand Open/Close, Pronation-Supination, Postural Tremor, and Rest
Tremor. Parallel to pose_test.py, but for a single tracked hand instead of
a full body.

Uses mediapipe's HandLandmarker (Tasks API, VIDEO mode, num_hands=1 -- one
hand per recording, per the brief) instead of PoseLandmarker. Deliberately
its own module, not folded into pose_test.py or pose_tracking.py:

1. A hand recording is a different shape of problem from a full-body one --
   one landmarker, one hand, framing/duration conventions that don't apply
   to the body tasks -- so keeping it separate avoids a single file trying
   to serve two different pipelines.
2. recording_quality.py is already video-agnostic by design (see its own
   docstring) and is reused here AS-IS, unmodified -- no body task existed
   when it was written, and no hand-specific fork of it was needed.
3. Every pure-logic piece below (peak/trough detection, the cyclic-signal
   summary, the FFT tremor summary) takes plain numbers/arrays, never a
   mediapipe object or an open video file, so it's fully unit-testable with
   synthetic signals of a known, hand-chosen count/frequency -- see
   tests/backend/test_hand_tasks.py.

HARD PRODUCT RULE for every summarize_* function below: return RAW
measurements only -- counts, times, angles in degrees, distances
normalized to the person's own hand size, frequencies in Hz, rates, and
variability (CV / std). Never a 0-100 score, a letter grade, a risk label,
or any diagnosis-like language. Kinetra is not validated against any
clinical reference yet, so none of that would mean anything defensible.
(Older body-task code in pose_test.py predates this rule and isn't being
retrofitted here -- this rule governs everything NEW in this module.)
"""

from __future__ import annotations

import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import mediapipe as mp

from recording_quality import RecordingQualityError, check_recording_quality

# Bumped whenever the hand detection/signal-processing pipeline changes in a
# way that could shift results for the same input video. Returned in every
# result, same convention as pose_test.py's ALGORITHM_VERSION, so
# validation data (and anyone comparing results over time) can tell which
# pipeline version produced a given number.
HAND_ALGORITHM_VERSION = "hand-v1-tasks-landmarker-2026.11"

# Same model family MediaPipe publishes for this exact task -- there's only
# one hand_landmarker weight class (no lite/full split like pose_landmarker
# has), so there's no size-vs-accuracy choice to make here.
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_CACHE_PATH = os.path.join(
    tempfile.gettempdir(), "kinetra_hand_landmarker.task"
)

# Render's free-tier instance is 512MB (see server.py's comment on this).
# server.py never holds a PoseTracker and a HandTracker in memory at the
# same time -- each route creates and closes its own landmarker per
# request -- specifically so this model and pose_landmarker_lite's never
# have to coexist in the same process's memory budget.

# HandLandmark indices -- confirmed by direct introspection against the
# installed mediapipe (mediapipe.python.solutions.hands.HandLandmark),
# identical topology under the Tasks API. Same technique used for
# pose_tracking.py's pose landmark indices.
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Per the brief: a per-frame hand detection below this confidence is treated
# as "missing" entirely (not used for the signal, not counted as visible),
# same spirit as a frame where mediapipe found nobody at all.
MIN_HAND_PRESENCE_CONFIDENCE = 0.9

FINGER_TAPPING_CAMERA = "finger_tapping_camera"
HAND_OPEN_CLOSE = "hand_open_close"
PRONATION_SUPINATION = "pronation_supination"
POSTURAL_TREMOR = "postural_tremor"
REST_TREMOR = "rest_tremor"

HAND_TASKS = (
    FINGER_TAPPING_CAMERA,
    HAND_OPEN_CLOSE,
    PRONATION_SUPINATION,
    POSTURAL_TREMOR,
    REST_TREMOR,
)

# The first three are "cyclic" tasks -- a repeated open/close-style motion,
# summarized by counting peaks and measuring their amplitude/rhythm. The
# last two are "tremor" tasks -- a held posture or rest position, summarized
# by the frequency spectrum of small involuntary position changes. Sharing
# one module lets both families reuse the same quality gate, model wrapper,
# and wrist-stability metric, even though their signal analysis differs.
CYCLIC_HAND_TASKS = (FINGER_TAPPING_CAMERA, HAND_OPEN_CLOSE, PRONATION_SUPINATION)
TREMOR_HAND_TASKS = (POSTURAL_TREMOR, REST_TREMOR)

# Tremor frequency band of interest, Hz. Chosen to cover both the classic
# "rest tremor" (~4-6Hz) and "postural/action tremor" (~6-12Hz) ranges
# without being so wide it picks up ordinary voluntary movement (<3Hz).
TREMOR_BAND_HZ: Tuple[float, float] = (3.0, 12.0)

# For best frequency resolution on a tremor task specifically (higher fps
# = finer frequency bins over the same recording length), 60fps is
# preferred even though recording_quality.py's general MIN_FPS stays 30.
TREMOR_PREFERRED_FPS = 60.0


def safe_float(x):
    return float(x) if x is not None else None


def ensure_model_downloaded(path: str = _MODEL_CACHE_PATH, url: str = HAND_MODEL_URL) -> str:
    """Download the .task model file once and cache it on disk. Mirrors
    pose_tracking.py's ensure_model_downloaded exactly -- see that
    function's docstring for why this isn't bundled in the repo or
    re-downloaded per request."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    tmp_path = path + ".partial"
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, path)
    return path


@dataclass
class HandFrameResult:
    landmarks: Optional[list]   # mediapipe landmark-like objects (x, y, z), or None
    confidence: float           # handedness classification score for the detected hand, 0 if none
    handedness: Optional[str]   # "Left" | "Right" | None

    @property
    def is_visible(self) -> bool:
        return self.landmarks is not None and self.confidence >= MIN_HAND_PRESENCE_CONFIDENCE


class HandTracker:
    """Thin wrapper around mediapipe's HandLandmarker (Tasks API, VIDEO
    mode, num_hands=1). Mirrors pose_tracking.py's PoseTracker shape
    (process()/close(), running totals read by the caller after the frame
    loop) deliberately, so run_hand_analysis()'s call site reads like
    pose_test.py's."""

    def __init__(self, model_path: Optional[str] = None):
        resolved_path = model_path or ensure_model_downloaded()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=resolved_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        # Running totals the caller (run_hand_analysis) folds into its
        # recording_quality.py check after the frame loop finishes -- same
        # convention as PoseTracker.low_confidence_frame_count/total_frame_count.
        self.low_confidence_frame_count = 0
        self.total_frame_count = 0

    def process(self, frame_rgb, timestamp_ms: int) -> HandFrameResult:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        self.total_frame_count += 1

        if not result.hand_landmarks:
            return HandFrameResult(landmarks=None, confidence=0.0, handedness=None)

        landmarks = result.hand_landmarks[0]
        confidence = 0.0
        handedness = None
        if result.handedness and result.handedness[0]:
            top = result.handedness[0][0]
            confidence = float(top.score)
            handedness = top.category_name

        if confidence < MIN_HAND_PRESENCE_CONFIDENCE:
            self.low_confidence_frame_count += 1

        return HandFrameResult(landmarks=landmarks, confidence=confidence, handedness=handedness)

    def close(self):
        self._landmarker.close()


# ---------------------------------------------------------------------------
# PURE GEOMETRY HELPERS -- plain (x, y) tuples in, plain floats out. Mirrors
# pose_test.py's calculate_angle() convention (callers extract .x/.y from
# landmark objects before calling, rather than these helpers knowing about
# mediapipe at all) so they're trivially unit-testable with fabricated
# points.
# ---------------------------------------------------------------------------

def _xy(landmark) -> Tuple[float, float]:
    return (landmark.x, landmark.y)


def _dist(a, b) -> float:
    ax, ay = _xy(a)
    bx, by = _xy(b)
    return float(math.hypot(ax - bx, ay - by))


def _angle_deg(vertex, point_a, point_b) -> float:
    """Angle at `vertex`, between the rays to point_a and point_b, in
    degrees (0-180). Used for the finger-tapping thumb-index angle."""
    vx, vy = _xy(vertex)
    ax, ay = _xy(point_a)
    bx, by = _xy(point_b)
    va = (ax - vx, ay - vy)
    vb = (bx - vx, by - vy)
    denom = math.hypot(*va) * math.hypot(*vb)
    if denom == 0:
        return 0.0
    cosine = (va[0] * vb[0] + va[1] * vb[1]) / denom
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _rotation_angle_rad(index_mcp, pinky_mcp) -> float:
    """Rotation of the hand about the forearm axis, estimated from the
    angle (radians, atan2 range) of the line across the knuckles --
    index-MCP to pinky-MCP -- relative to image-horizontal. As the hand
    pronates/supinates, this line sweeps through roughly a half-turn.
    Returned in radians (not degrees) so the per-frame signal can be
    np.unwrap()'d before use -- atan2 wraps at +-180 deg, which would
    otherwise look like a huge, spurious one-frame "movement" to the peak
    detector exactly when the hand happens to cross that wrap point."""
    ix, iy = _xy(index_mcp)
    px, py = _xy(pinky_mcp)
    return math.atan2(py - iy, px - ix)


# ---------------------------------------------------------------------------
# PEAK / TROUGH DETECTION -- a minimal, dependency-light local-extrema
# finder (no scipy -- see requirements.txt's own comments on why new
# transitive dependencies here get pinned deliberately, not added
# casually). Good enough for the roughly-periodic, roughly-sinusoidal
# signals these five tasks produce; tuned and verified against synthetic
# signals of a known count/amplitude in tests/backend/test_hand_tasks.py.
# ---------------------------------------------------------------------------

def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(signal) < 2:
        return signal.astype(float)
    window = window if window % 2 == 1 else window + 1  # keep it centered
    kernel = np.ones(window) / window
    # 'same'-mode convolution with edge-padding so the output stays the
    # same length and doesn't dim toward zero at the start/end.
    padded = np.pad(signal, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _raw_local_extrema(signal: np.ndarray) -> Tuple[List[int], List[int]]:
    peaks, troughs = [], []
    for i in range(1, len(signal) - 1):
        if signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]:
            peaks.append(i)
        elif signal[i] <= signal[i - 1] and signal[i] < signal[i + 1]:
            troughs.append(i)
    return peaks, troughs


def _filter_extrema(
    signal: np.ndarray, candidates: List[int], min_distance: int, min_prominence: float, kind: str
) -> List[int]:
    """Keep only extrema that are at least `min_distance` samples apart and
    at least `min_prominence` tall/deep relative to their immediate
    surroundings -- drops sensor-jitter "extrema" that aren't a real
    tap/cycle."""
    kept: List[int] = []
    n = len(signal)
    for idx in candidates:
        lo, hi = max(0, idx - min_distance), min(n, idx + min_distance + 1)
        if kind == "peak":
            base = min(np.min(signal[lo:idx + 1]), np.min(signal[idx:hi]))
            prominence = signal[idx] - base
        else:
            base = max(np.max(signal[lo:idx + 1]), np.max(signal[idx:hi]))
            prominence = base - signal[idx]
        if prominence < min_prominence:
            continue
        if kept and (idx - kept[-1]) < min_distance:
            better = (
                signal[idx] > signal[kept[-1]] if kind == "peak"
                else signal[idx] < signal[kept[-1]]
            )
            if better:
                kept[-1] = idx
            continue
        kept.append(idx)
    return kept


def find_peaks_and_troughs(
    signal: Sequence[float], min_distance: int, prominence_ratio: float = 0.1
) -> Tuple[List[int], List[int]]:
    """Find local maxima/minima in `signal`, filtered by a minimum sample
    spacing and a minimum prominence (as a fraction of the signal's own
    peak-to-peak range, so it self-scales to whatever units the caller's
    signal is in -- degrees, a normalized distance ratio, radians)."""
    arr = np.asarray(signal, dtype=float)
    if len(arr) < 3:
        return [], []
    signal_range = float(np.max(arr) - np.min(arr))
    min_prominence = prominence_ratio * signal_range if signal_range > 0 else 0.0
    raw_peaks, raw_troughs = _raw_local_extrema(arr)
    peaks = _filter_extrema(arr, raw_peaks, min_distance, min_prominence, "peak")
    troughs = _filter_extrema(arr, raw_troughs, min_distance, min_prominence, "trough")
    return peaks, troughs


# ---------------------------------------------------------------------------
# CYCLIC SIGNAL SUMMARY -- shared by Finger Tapping (Camera), Hand
# Open/Close, and Pronation-Supination. Each task builds a different raw
# per-frame signal (see run_hand_analysis below) but all three are
# "something opens/rotates and comes back, repeatedly" -- counted, timed,
# and measured for amplitude/rhythm the same way.
# ---------------------------------------------------------------------------

@dataclass
class CyclicMetrics:
    cycle_count: int
    cycle_rate_hz: Optional[float]
    mean_amplitude: Optional[float]
    max_amplitude: Optional[float]
    amplitude_decrement_percent: Optional[float]
    mean_cycle_speed: Optional[float]
    rhythm_cv: Optional[float]
    halt_count: int


def analyze_cyclic_signal(
    signal: Sequence[float], timestamps: Sequence[float], fps: float
) -> CyclicMetrics:
    """Count repeated open/rotate cycles in `signal` and summarize their
    amplitude, speed, rhythm, and any halts (pauses much longer than the
    established rhythm). `timestamps` are the per-sample time in seconds
    (same length as `signal`) -- used rather than a flat 1/fps step so a
    handful of low-confidence frames dropped mid-recording (see
    run_hand_analysis) don't silently compress the apparent timing.
    """
    arr = np.asarray(signal, dtype=float)
    if len(arr) < 5:
        return CyclicMetrics(0, None, None, None, None, None, None, 0)

    smooth_window = max(3, int(round(fps * 0.05)))
    smooth = _moving_average(arr, smooth_window)

    # Fastest plausible voluntary tap/cycle rate is well under ~6-7Hz,
    # i.e. a period no shorter than ~0.14-0.17s -- used as the minimum
    # spacing between accepted peaks so one tap can't be double-counted
    # from a noisy frame or two near its apex.
    min_distance = max(2, int(round(fps * 0.12)))
    peaks, troughs = find_peaks_and_troughs(smooth, min_distance=min_distance, prominence_ratio=0.1)

    if not peaks:
        return CyclicMetrics(0, None, None, None, None, None, None, 0)

    amplitudes: List[float] = []
    speeds: List[float] = []
    for p in peaks:
        preceding = [t for t in troughs if t < p]
        if preceding:
            base_idx = preceding[-1]
            base_value = float(smooth[base_idx])
        else:
            base_idx = int(np.argmin(smooth[:p + 1]))
            base_value = float(smooth[base_idx])
        amplitudes.append(float(smooth[p]) - base_value)

        dt = timestamps[p] - timestamps[base_idx] if base_idx != p else None
        if dt and dt > 0:
            speeds.append((float(smooth[p]) - base_value) / dt)

    cycle_count = len(peaks)
    duration_seconds = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else (len(arr) / fps if fps else None)
    cycle_rate_hz = (cycle_count / duration_seconds) if duration_seconds and duration_seconds > 0 else None

    mean_amplitude = float(np.mean(amplitudes))
    max_amplitude = float(np.max(amplitudes))

    if cycle_count >= 6:
        first_three = float(np.mean(amplitudes[:3]))
        last_three = float(np.mean(amplitudes[-3:]))
        amplitude_decrement_percent = (
            ((first_three - last_three) / first_three) * 100.0 if first_three != 0 else None
        )
    else:
        amplitude_decrement_percent = None

    mean_cycle_speed = float(np.mean(speeds)) if speeds else None

    if cycle_count >= 3:
        peak_times = [timestamps[p] for p in peaks]
        intervals = np.diff(peak_times)
        mean_interval = float(np.mean(intervals))
        rhythm_cv = float(np.std(intervals) / mean_interval) if mean_interval > 0 else None
        median_interval = float(np.median(intervals))
        halt_count = int(np.sum(intervals > 2.0 * median_interval)) if median_interval > 0 else 0
    else:
        rhythm_cv = None
        halt_count = 0

    return CyclicMetrics(
        cycle_count=cycle_count,
        cycle_rate_hz=safe_float(cycle_rate_hz),
        mean_amplitude=safe_float(mean_amplitude),
        max_amplitude=safe_float(max_amplitude),
        amplitude_decrement_percent=safe_float(amplitude_decrement_percent),
        mean_cycle_speed=safe_float(mean_cycle_speed),
        rhythm_cv=safe_float(rhythm_cv),
        halt_count=halt_count,
    )


# ---------------------------------------------------------------------------
# TREMOR SIGNAL SUMMARY -- shared by Postural Tremor and Rest Tremor.
# ---------------------------------------------------------------------------

@dataclass
class TremorMetrics:
    dominant_frequency_hz: Optional[float]
    power_in_tremor_band_fraction: Optional[float]
    relative_amplitude: Optional[float]


def _power_spectrum(signal: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    detrended = signal - np.mean(signal)
    window = np.hanning(n) if n > 1 else np.ones(n)
    spectrum = np.fft.rfft(detrended * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(spectrum) ** 2
    return freqs, power


def analyze_tremor_signal_2d(
    x_signal: Sequence[float], y_signal: Sequence[float], fps: float, hand_length: Optional[float]
) -> TremorMetrics:
    """FFT-based tremor summary from a 2D fingertip position time series
    (x_signal, y_signal -- same length, same units, normalized 0-1 image
    coordinates). Both axes are analyzed and combined rather than just one,
    since a tremor's dominant axis isn't known ahead of time and isn't
    guaranteed to be purely horizontal or vertical relative to the camera.

    `hand_length` is a per-recording distance (wrist-to-middle-fingertip,
    see run_hand_analysis) used only to express amplitude as a fraction of
    the person's own hand size -- never claimed as a real-world distance,
    since there's no scale reference (ruler, known object) in frame.
    """
    x = np.asarray(x_signal, dtype=float)
    y = np.asarray(y_signal, dtype=float)
    n = min(len(x), len(y))

    # Need at least ~1 second of samples for the lowest tremor-band
    # frequency (3Hz) to even complete a single cycle within the window.
    if n == 0 or fps <= 0 or n < int(fps * 1.0):
        return TremorMetrics(None, None, None)

    x, y = x[:n], y[:n]
    freqs, power_x = _power_spectrum(x, fps)
    _, power_y = _power_spectrum(y, fps)
    power = power_x + power_y

    band_mask = (freqs >= TREMOR_BAND_HZ[0]) & (freqs <= TREMOR_BAND_HZ[1])
    total_power = float(np.sum(power))
    power_in_band = float(np.sum(power[band_mask])) if np.any(band_mask) else 0.0
    power_in_tremor_band_fraction = (power_in_band / total_power) if total_power > 0 else None

    dominant_frequency_hz = None
    if np.any(band_mask) and np.max(power[band_mask]) > 0:
        band_freqs = freqs[band_mask]
        band_power = power[band_mask]
        dominant_frequency_hz = float(band_freqs[int(np.argmax(band_power))])

    relative_amplitude = None
    if hand_length:
        radial_std = float(math.hypot(float(np.std(x)), float(np.std(y))))
        relative_amplitude = radial_std / hand_length

    return TremorMetrics(
        dominant_frequency_hz=safe_float(dominant_frequency_hz),
        power_in_tremor_band_fraction=safe_float(power_in_tremor_band_fraction),
        relative_amplitude=safe_float(relative_amplitude),
    )


def _wrist_stability(wrist_positions: Sequence[Tuple[float, float]], hand_length: Optional[float]) -> Optional[float]:
    """How much the wrist itself drifted over the recording (as opposed to
    the fingers/hand moving relative to the wrist), as a fraction of hand
    length. High wrist drift during a finger-only task (tapping, open/
    close, pronation-supination) usually means the whole arm/forearm was
    moving, not just the fingers/hand -- useful context for interpreting
    the rest of that task's metrics, reported raw rather than judged."""
    if len(wrist_positions) < 3 or not hand_length:
        return None
    arr = np.asarray(wrist_positions, dtype=float)
    centered = arr - np.mean(arr, axis=0)
    radial = np.sqrt(np.sum(centered ** 2, axis=1))
    return float(np.std(radial) / hand_length)


# ---------------------------------------------------------------------------
# PER-TASK METRIC PACKAGING -- names, units, and a one-line plain-English
# description for every metric, per the brief's "every new metric needs a
# units label and a plain-English one-line description in the UI" rule.
# Carried from the backend rather than duplicated in the frontend, so the
# two can never drift out of sync with each other.
# ---------------------------------------------------------------------------

def _package_cyclic_metrics(hand_task: str, metrics: CyclicMetrics) -> Dict[str, Any]:
    if metrics.cycle_count == 0:
        return {
            "task_type": hand_task,
            "status": "insufficient_hand_data",
        }

    if hand_task == FINGER_TAPPING_CAMERA:
        count_key, rate_key = "tap_count", "tap_rate_hz"
        amp_unit, speed_unit = "deg", "deg/s"
        count_desc = "Number of thumb-to-index taps detected."
        rate_desc = "Taps per second, averaged over the whole recording."
        amp_mean_desc = "Average distance the thumb and index finger opened apart on each tap, in degrees of the angle at the wrist."
        amp_max_desc = "The single widest thumb-index opening across all taps, in degrees."
        speed_desc = "Average speed of opening from pinched to fully open, in degrees per second."
    elif hand_task == HAND_OPEN_CLOSE:
        count_key, rate_key = "cycle_count", "cycle_rate_hz"
        amp_unit, speed_unit = "ratio (fraction of palm size)", "ratio/s"
        count_desc = "Number of full hand open-close cycles detected."
        rate_desc = "Open-close cycles per second, averaged over the whole recording."
        amp_mean_desc = "Average how far the fingertips moved from the wrist on each cycle, as a fraction of palm size (so it's comparable across hand sizes/camera distances)."
        amp_max_desc = "The single largest hand-opening distance across all cycles, as a fraction of palm size."
        speed_desc = "Average speed of opening the hand, in palm-size fractions per second."
    else:  # PRONATION_SUPINATION
        count_key, rate_key = "cycle_count", "cycle_rate_hz"
        amp_unit, speed_unit = "deg", "deg/s"
        count_desc = "Number of palm-turning (pronation-supination) cycles detected."
        rate_desc = "Palm-turning cycles per second, averaged over the whole recording."
        amp_mean_desc = "Average rotation of the palm on each turn, in degrees."
        amp_max_desc = "The single largest palm rotation across all cycles, in degrees."
        speed_desc = "Average speed of turning the palm, in degrees per second."

    return {
        "task_type": hand_task,
        "status": f"{hand_task}_v1_analysis",
        count_key: metrics.cycle_count,
        rate_key: metrics.cycle_rate_hz,
        "mean_amplitude": metrics.mean_amplitude,
        "max_amplitude": metrics.max_amplitude,
        "amplitude_decrement_percent": metrics.amplitude_decrement_percent,
        "mean_cycle_speed": metrics.mean_cycle_speed,
        "rhythm_cv": metrics.rhythm_cv,
        "halt_count": metrics.halt_count,
        "metric_units": {
            count_key: "count",
            rate_key: "Hz",
            "mean_amplitude": amp_unit,
            "max_amplitude": amp_unit,
            "amplitude_decrement_percent": "%",
            "mean_cycle_speed": speed_unit,
            "rhythm_cv": "coefficient of variation (unitless)",
            "halt_count": "count",
        },
        "metric_descriptions": {
            count_key: count_desc,
            rate_key: rate_desc,
            "mean_amplitude": amp_mean_desc,
            "max_amplitude": amp_max_desc,
            "amplitude_decrement_percent": "How much smaller the last 3 movements were than the first 3, as a percent. A positive number means movements got smaller over time.",
            "mean_cycle_speed": speed_desc,
            "rhythm_cv": "How evenly spaced the movements were in time. Higher means less regular timing.",
            "halt_count": "Number of pauses in the movement that were much longer than the established rhythm.",
        },
    }


def _package_tremor_metrics(hand_task: str, metrics: TremorMetrics) -> Dict[str, Any]:
    if metrics.dominant_frequency_hz is None and metrics.relative_amplitude is None:
        return {
            "task_type": hand_task,
            "status": "insufficient_hand_data",
        }

    return {
        "task_type": hand_task,
        "status": f"{hand_task}_v1_analysis",
        "dominant_frequency_hz": metrics.dominant_frequency_hz,
        "power_in_tremor_band_fraction": metrics.power_in_tremor_band_fraction,
        "relative_amplitude": metrics.relative_amplitude,
        "metric_units": {
            "dominant_frequency_hz": "Hz",
            "power_in_tremor_band_fraction": "fraction of total signal power (0-1)",
            "relative_amplitude": "fraction of hand length (unitless)",
        },
        "metric_descriptions": {
            "dominant_frequency_hz": "The strongest repeating frequency found in the hand's small position changes, in cycles per second, within the 3-12Hz range typical of tremor.",
            "power_in_tremor_band_fraction": "How much of the hand's overall movement, during this recording, fell within the 3-12Hz tremor range versus slower/faster movement.",
            "relative_amplitude": "The size of the hand's small oscillations, as a fraction of the hand's own length (never an absolute distance -- there's no ruler in frame to calibrate to centimeters).",
        },
    }


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run_hand_analysis(
    video_path: str,
    hand_task: str = FINGER_TAPPING_CAMERA,
    hand: str = "right",
    trial_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze one recording of one of the five hand tasks. Raises
    RecordingQualityError (from recording_quality.py, unmodified) if the
    recording fails a hard quality bar; raises RuntimeError for an
    unreadable video or an unrecognized hand_task."""
    if hand_task not in HAND_TASKS:
        raise RuntimeError(f"Unknown hand_task '{hand_task}'. Must be one of {HAND_TASKS}.")

    normalized_hand = str(hand).strip().lower() if hand else "right"
    if normalized_hand not in ("left", "right"):
        normalized_hand = "right"

    import cv2  # local import mirrors pose_test.py's lazy-import style

    tracker = HandTracker()
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps == 0:
            fps = 30.0
        fps = float(fps)
        # Same sanity clamp as pose_test.py's run_analysis -- guards against
        # bogus container metadata from VFR-encoded phone recordings.
        if not (1.0 <= fps <= 120.0):
            fps = 30.0

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_index = 0
        tracker_last_timestamp_ms = -1
        total_frames = 0
        visible_frame_count = 0

        raw_signal: List[float] = []       # primary per-visible-frame signal (task-specific)
        secondary_signal: List[float] = []  # tremor tasks only: the y-axis companion to raw_signal's x
        wrist_positions: List[Tuple[float, float]] = []
        hand_lengths: List[float] = []
        timestamps: List[float] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((frame_index / fps) * 1000) if fps else frame_index * 33
            if timestamp_ms <= tracker_last_timestamp_ms:
                timestamp_ms = tracker_last_timestamp_ms + 1
            tracker_last_timestamp_ms = timestamp_ms
            frame_index += 1

            hand_result = tracker.process(image_rgb, timestamp_ms)

            if hand_result.is_visible:
                visible_frame_count += 1
                lm = hand_result.landmarks
                t_seconds = timestamp_ms / 1000.0
                timestamps.append(t_seconds)
                wrist_positions.append(_xy(lm[WRIST]))
                hand_lengths.append(_dist(lm[WRIST], lm[MIDDLE_FINGER_TIP]))

                if hand_task == FINGER_TAPPING_CAMERA:
                    raw_signal.append(_angle_deg(lm[WRIST], lm[THUMB_TIP], lm[INDEX_FINGER_TIP]))

                elif hand_task == HAND_OPEN_CLOSE:
                    palm_size = _dist(lm[WRIST], lm[MIDDLE_FINGER_MCP]) or 1e-6
                    tip_distances = [
                        _dist(lm[WRIST], lm[tip_idx])
                        for tip_idx in (INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP)
                    ]
                    raw_signal.append(float(np.mean(tip_distances)) / palm_size)

                elif hand_task == PRONATION_SUPINATION:
                    raw_signal.append(_rotation_angle_rad(lm[INDEX_FINGER_MCP], lm[PINKY_MCP]))

                else:  # POSTURAL_TREMOR / REST_TREMOR
                    # Tremor is a small involuntary position change, not a
                    # deliberate angle -- so the fingertip's own (x, y)
                    # position is the signal, not a derived angle/distance.
                    # The middle fingertip is used since it's the most
                    # distal, least self-occluded point on an open hand.
                    raw_signal.append(lm[MIDDLE_FINGER_TIP].x)
                    secondary_signal.append(lm[MIDDLE_FINGER_TIP].y)

        quality = check_recording_quality(
            total_frames=total_frames,
            visible_frame_count=visible_frame_count,
            fps=fps,
            low_confidence_tracking_count=tracker.low_confidence_frame_count,
        )
        if not quality.passed:
            raise RecordingQualityError(quality.reasons)

        warnings = list(quality.warnings)
        if hand_task in TREMOR_HAND_TASKS and fps < TREMOR_PREFERRED_FPS:
            warnings.append(
                f"This recording was analyzed at {fps:.0f}fps. For the clearest tremor-frequency "
                f"reading, {TREMOR_PREFERRED_FPS:.0f}fps is preferred if your device supports it."
            )

        hand_length = float(np.median(hand_lengths)) if hand_lengths else None

        if hand_task == PRONATION_SUPINATION and raw_signal:
            # Unwrap now, after collection -- atan2's +-180deg wrap would
            # otherwise look like a huge spurious one-frame "rotation" to
            # the peak detector exactly when the hand crosses that angle.
            raw_signal = list(np.degrees(np.unwrap(raw_signal)))

        if hand_task in CYCLIC_HAND_TASKS:
            metrics = analyze_cyclic_signal(raw_signal, timestamps, fps)
            task_analysis = _package_cyclic_metrics(hand_task, metrics)
        else:
            tremor_metrics = analyze_tremor_signal_2d(raw_signal, secondary_signal, fps, hand_length)
            task_analysis = _package_tremor_metrics(hand_task, tremor_metrics)

        if task_analysis.get("status") != "insufficient_hand_data":
            wrist_stability = _wrist_stability(wrist_positions, hand_length)
            task_analysis["wrist_stability"] = safe_float(wrist_stability)
            task_analysis.setdefault("metric_units", {})["wrist_stability"] = "fraction of hand length (unitless)"
            task_analysis.setdefault("metric_descriptions", {})["wrist_stability"] = (
                "How much the wrist itself drifted during the recording, as a fraction of hand length. "
                "High drift usually means the whole hand/arm was moving, not just the fingers."
            )

        result_data: Dict[str, Any] = {
            "algorithm_version": HAND_ALGORITHM_VERSION,
            "hand_task": hand_task,
            "hand": normalized_hand,
            "trial_index": trial_index,
            "total_frames": total_frames,
            "fps": fps,
            "recording_quality_warnings": warnings,
            # Same key names as pose_test.py's result_data -- raw counts,
            # not ratios, so a dashboard/frontend reading either task
            # family's result doesn't need two different shapes.
            "tracking_confidence_summary": {
                "low_confidence_frames": tracker.low_confidence_frame_count,
                "total_frames_tracked": tracker.total_frame_count,
            },
            "visible_frame_ratio": quality.visible_frame_ratio,
            "task_analysis": task_analysis,
        }

        return result_data
    finally:
        if cap is not None:
            cap.release()
        tracker.close()


if __name__ == "__main__":
    import sys

    _video_path = os.getenv("VIDEO_PATH", sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4")
    _hand_task = os.getenv("HAND_TASK", FINGER_TAPPING_CAMERA)
    _hand = os.getenv("HAND", "right")

    import json as _json
    print(_json.dumps(run_hand_analysis(_video_path, hand_task=_hand_task, hand=_hand), indent=2))
