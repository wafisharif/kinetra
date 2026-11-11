"""Unit + integration tests for hand_tasks.py (Phase 1: Finger Tapping
(Camera), Hand Open/Close, Pronation-Supination, Postural Tremor, Rest
Tremor).

Mirrors the existing test split for the Phase 0 body-task pipeline:
- Pure-logic pieces (peak/trough detection, the cyclic-signal summary, the
  FFT tremor summary, wrist stability) are tested with fabricated/synthetic
  signals of a known, hand-chosen count/amplitude/frequency -- no video, no
  real mediapipe model, no network required. See test_pose_tracking.py and
  test_recording_quality.py for the same approach applied to Phase 0.
- run_hand_analysis() itself is tested end-to-end with a real (tiny,
  synthetic) video file but a stand-in HandTracker, so the WIRING between
  video reading, the tracker, recording_quality.py, and the per-task
  summarizers is covered without needing a real downloaded .task model --
  see test_run_analysis_pose_integration.py for the same approach applied
  to pose_test.py's run_analysis().
"""
import math
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import hand_tasks
from hand_tasks import (
    CyclicMetrics,
    HandFrameResult,
    TremorMetrics,
    analyze_cyclic_signal,
    analyze_tremor_signal_2d,
    find_peaks_and_troughs,
)
from recording_quality import RecordingQualityError


# --------------------------------------------------------------------------
# find_peaks_and_troughs -- pure geometry/signal logic, no mediapipe
# --------------------------------------------------------------------------

def _sine(freq_hz, duration_s, fps, amplitude=1.0, offset=0.0, phase=-math.pi / 2):
    n = int(fps * duration_s)
    t = np.arange(n) / fps
    return t, offset + amplitude * np.sin(2 * math.pi * freq_hz * t + phase)


def test_find_peaks_counts_a_clean_sine_correctly():
    # phase=0 (not the default -pi/2) so neither a peak nor a trough lands
    # exactly on sample 0 or the last sample -- an extremum tied to a
    # boundary sample is an edge artifact of the test construction itself
    # (there's no "previous"/"next" sample to compare against), not
    # something the detector should be expected to find.
    _, signal = _sine(freq_hz=2.0, duration_s=10.0, fps=30.0, amplitude=30.0, offset=50.0, phase=0.0)
    peaks, troughs = find_peaks_and_troughs(signal, min_distance=3)
    assert len(peaks) == 20  # 2Hz over 10s
    assert len(troughs) == 20


def test_find_peaks_ignores_tiny_jitter_below_prominence_floor():
    _, base = _sine(freq_hz=1.0, duration_s=5.0, fps=30.0, amplitude=40.0, offset=50.0)
    rng = np.random.default_rng(0)
    jitter = base + rng.normal(scale=0.5, size=len(base))  # tiny noise, well under 10% of range
    peaks, _ = find_peaks_and_troughs(jitter, min_distance=3, prominence_ratio=0.1)
    assert len(peaks) == 5  # still exactly 5 real cycles, jitter didn't add spurious peaks


def test_find_peaks_on_short_signal_returns_empty():
    peaks, troughs = find_peaks_and_troughs([1.0, 2.0], min_distance=3)
    assert peaks == []
    assert troughs == []


# --------------------------------------------------------------------------
# analyze_cyclic_signal -- shared by Finger Tapping, Hand Open/Close,
# Pronation-Supination
# --------------------------------------------------------------------------

def test_analyze_cyclic_signal_regular_tapping():
    fps = 30.0
    t, signal = _sine(freq_hz=2.0, duration_s=10.0, fps=fps, amplitude=30.0, offset=50.0)
    metrics = analyze_cyclic_signal(signal.tolist(), t.tolist(), fps)

    assert metrics.cycle_count == 20
    assert metrics.cycle_rate_hz == pytest.approx(2.0, abs=0.05)
    assert metrics.mean_amplitude == pytest.approx(60.0, rel=0.15)  # peak-trough ~= 2*amplitude
    assert metrics.rhythm_cv < 0.1  # near-perfectly regular
    assert metrics.halt_count == 0
    assert metrics.amplitude_decrement_percent == pytest.approx(0.0, abs=5.0)


def test_analyze_cyclic_signal_detects_amplitude_decrement():
    fps = 30.0
    n = int(fps * 10.0)
    t = np.arange(n) / fps
    envelope = np.linspace(35.0, 10.0, n)  # shrinking amplitude -- classic decrement pattern
    signal = 50.0 + envelope * np.sin(2 * math.pi * 2.0 * t - math.pi / 2)

    metrics = analyze_cyclic_signal(signal.tolist(), t.tolist(), fps)

    assert metrics.cycle_count > 0
    assert metrics.amplitude_decrement_percent > 30.0  # clearly decrementing, not noise-level


def test_analyze_cyclic_signal_detects_a_halt():
    fps = 30.0
    t, signal = _sine(freq_hz=2.0, duration_s=10.0, fps=fps, amplitude=30.0, offset=50.0)
    signal = signal.copy()
    halt_start, halt_end = int(4 * fps), int(6 * fps)
    signal[halt_start:halt_end] = signal[halt_start]  # flatten a 2-second window: a freeze

    metrics = analyze_cyclic_signal(signal.tolist(), t.tolist(), fps)

    assert metrics.halt_count >= 1
    assert metrics.cycle_count < 20  # fewer cycles than the uninterrupted version


def test_analyze_cyclic_signal_too_short_returns_zero_cycles():
    metrics = analyze_cyclic_signal([1.0, 2.0, 3.0], [0.0, 0.03, 0.06], fps=30.0)
    assert metrics.cycle_count == 0
    assert metrics.mean_amplitude is None


# --------------------------------------------------------------------------
# analyze_tremor_signal_2d -- shared by Postural Tremor, Rest Tremor
# --------------------------------------------------------------------------

def test_analyze_tremor_signal_2d_detects_known_frequency():
    fps = 60.0
    duration = 3.0
    n = int(fps * duration)
    t = np.arange(n) / fps
    x = 0.5 + 0.01 * np.sin(2 * math.pi * 5.0 * t)
    y = 0.5 + 0.008 * np.cos(2 * math.pi * 5.0 * t)

    metrics = analyze_tremor_signal_2d(x.tolist(), y.tolist(), fps, hand_length=0.15)

    assert metrics.dominant_frequency_hz == pytest.approx(5.0, abs=0.5)
    assert metrics.power_in_tremor_band_fraction > 0.9  # pure in-band sine, almost all power in-band
    assert metrics.relative_amplitude is not None
    assert metrics.relative_amplitude > 0


def test_analyze_tremor_signal_2d_outside_band_has_low_band_power():
    fps = 60.0
    duration = 3.0
    n = int(fps * duration)
    t = np.arange(n) / fps
    # 1Hz is a normal voluntary-movement rate, well below the 3-12Hz tremor band.
    x = 0.5 + 0.02 * np.sin(2 * math.pi * 1.0 * t)
    y = np.full(n, 0.5)

    metrics = analyze_tremor_signal_2d(x.tolist(), y.tolist(), fps, hand_length=0.15)

    assert metrics.power_in_tremor_band_fraction < 0.3


def test_analyze_tremor_signal_2d_insufficient_samples_returns_none():
    metrics = analyze_tremor_signal_2d([0.5, 0.5], [0.5, 0.5], fps=30.0, hand_length=0.15)
    assert metrics.dominant_frequency_hz is None
    assert metrics.relative_amplitude is None


# --------------------------------------------------------------------------
# _wrist_stability
# --------------------------------------------------------------------------

def test_wrist_stability_zero_for_a_perfectly_still_wrist():
    positions = [(0.5, 0.5)] * 30
    assert hand_tasks._wrist_stability(positions, hand_length=0.15) == 0.0


def test_wrist_stability_positive_when_wrist_drifts():
    rng = np.random.default_rng(1)
    positions = [(0.5 + rng.normal(scale=0.02), 0.5 + rng.normal(scale=0.02)) for _ in range(30)]
    stability = hand_tasks._wrist_stability(positions, hand_length=0.15)
    assert stability > 0


def test_wrist_stability_none_without_hand_length():
    positions = [(0.5, 0.5)] * 10
    assert hand_tasks._wrist_stability(positions, hand_length=None) is None


# --------------------------------------------------------------------------
# Metric packaging -- every metric must carry a unit and a plain-English
# description (hard product rule from the brief), and nothing resembling a
# 0-100 score or letter grade.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("task", hand_tasks.CYCLIC_HAND_TASKS)
def test_cyclic_metric_packaging_has_units_and_descriptions_for_every_metric(task):
    metrics = CyclicMetrics(
        cycle_count=10, cycle_rate_hz=2.0, mean_amplitude=40.0, max_amplitude=55.0,
        amplitude_decrement_percent=12.0, mean_cycle_speed=90.0, rhythm_cv=0.08, halt_count=0,
    )
    packaged = hand_tasks._package_cyclic_metrics(task, metrics)

    numeric_metric_keys = [
        k for k, v in packaged.items()
        if k not in ("task_type", "status", "metric_units", "metric_descriptions")
    ]
    assert len(numeric_metric_keys) >= 7
    for key in numeric_metric_keys:
        assert key in packaged["metric_units"], f"{key} missing a unit"
        assert key in packaged["metric_descriptions"], f"{key} missing a description"
        assert len(packaged["metric_descriptions"][key]) > 10

    # No score/grade language anywhere in this new feature's output.
    forbidden_substrings = ("score", "grade", "excellent", "moderate", "needs work", "risk")
    blob = " ".join(packaged["metric_descriptions"].values()).lower()
    assert not any(bad in blob for bad in forbidden_substrings)


@pytest.mark.parametrize("task", hand_tasks.TREMOR_HAND_TASKS)
def test_tremor_metric_packaging_has_units_and_descriptions_for_every_metric(task):
    metrics = TremorMetrics(
        dominant_frequency_hz=5.2, power_in_tremor_band_fraction=0.7, relative_amplitude=0.04,
    )
    packaged = hand_tasks._package_tremor_metrics(task, metrics)

    for key in ("dominant_frequency_hz", "power_in_tremor_band_fraction", "relative_amplitude"):
        assert key in packaged["metric_units"]
        assert key in packaged["metric_descriptions"]


def test_cyclic_metric_packaging_insufficient_data_short_circuits():
    packaged = hand_tasks._package_cyclic_metrics(
        hand_tasks.FINGER_TAPPING_CAMERA,
        CyclicMetrics(0, None, None, None, None, None, None, 0),
    )
    assert packaged == {"task_type": "finger_tapping_camera", "status": "insufficient_hand_data"}


# --------------------------------------------------------------------------
# run_hand_analysis -- integration, with a stand-in HandTracker (no real
# model download/inference -- see the plan doc for why that's still
# pending a real .task model file reachable from this environment).
# --------------------------------------------------------------------------

FRAME_COUNT = 300
FPS = 30.0


class _FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def _tapping_landmarks(frame_index: int) -> list:
    """21 landmarks shaped so the thumb-index angle at the wrist oscillates
    at 2Hz, amplitude ~30deg around a 50deg baseline -- same signal shape
    already hand-verified against analyze_cyclic_signal directly."""
    landmarks = [_FakeLandmark(0.5, 0.5) for _ in range(21)]
    wrist = (0.5, 0.8)
    landmarks[hand_tasks.WRIST] = _FakeLandmark(*wrist)
    landmarks[hand_tasks.MIDDLE_FINGER_MCP] = _FakeLandmark(0.5, 0.6)  # palm size = 0.2

    t = frame_index / FPS
    target_angle_deg = 50.0 + 30.0 * math.sin(2 * math.pi * 2.0 * t - math.pi / 2)
    r = 0.1
    theta_index = math.radians(-90.0)
    theta_thumb = theta_index + math.radians(target_angle_deg)
    landmarks[hand_tasks.INDEX_FINGER_TIP] = _FakeLandmark(
        wrist[0] + r * math.cos(theta_index), wrist[1] + r * math.sin(theta_index)
    )
    landmarks[hand_tasks.THUMB_TIP] = _FakeLandmark(
        wrist[0] + r * math.cos(theta_thumb), wrist[1] + r * math.sin(theta_thumb)
    )
    return landmarks


def _tremor_landmarks(frame_index: int) -> list:
    """21 landmarks shaped so the middle fingertip oscillates at 5Hz with a
    small amplitude -- a classic tremor signal."""
    landmarks = [_FakeLandmark(0.5, 0.5) for _ in range(21)]
    landmarks[hand_tasks.WRIST] = _FakeLandmark(0.5, 0.8)

    t = frame_index / FPS
    x = 0.5 + 0.01 * math.sin(2 * math.pi * 5.0 * t)
    y = 0.5 - 0.008 * math.cos(2 * math.pi * 5.0 * t)
    landmarks[hand_tasks.MIDDLE_FINGER_TIP] = _FakeLandmark(x, y)
    return landmarks


class _FakeHandTracker:
    """Stand-in for hand_tasks.HandTracker: same shape (process/close + the
    two running counters run_hand_analysis reads), but returns canned,
    controllable results per frame instead of running real inference."""

    def __init__(self, landmarks_fn, visible_frame_ratio=1.0, low_confidence_ratio=0.0):
        self.landmarks_fn = landmarks_fn
        self.visible_frame_ratio = visible_frame_ratio
        self.low_confidence_ratio = low_confidence_ratio
        self.low_confidence_frame_count = 0
        self.total_frame_count = 0

    def process(self, frame_rgb, timestamp_ms):
        frame_index = self.total_frame_count
        self.total_frame_count += 1
        is_visible = (frame_index % 100) < (self.visible_frame_ratio * 100)
        is_low_confidence = (frame_index % 100) < (self.low_confidence_ratio * 100)
        if is_low_confidence:
            self.low_confidence_frame_count += 1
        if not is_visible:
            return HandFrameResult(landmarks=None, confidence=0.0, handedness=None)
        return HandFrameResult(
            landmarks=self.landmarks_fn(frame_index), confidence=0.98, handedness="Right"
        )

    def close(self):
        pass


def _write_synthetic_video(path: str, frame_count: int = FRAME_COUNT, fps: float = FPS):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for i in range(frame_count):
        frame = np.full((64, 64, 3), fill_value=(i % 255), dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_video(tmp_path):
    video_path = str(tmp_path / "synthetic_hand.mp4")
    _write_synthetic_video(video_path)
    return video_path


def test_run_hand_analysis_finger_tapping_end_to_end(synthetic_video):
    fake_tracker = _FakeHandTracker(_tapping_landmarks)
    with patch("hand_tasks.HandTracker", return_value=fake_tracker):
        result = hand_tasks.run_hand_analysis(
            video_path=synthetic_video,
            hand_task=hand_tasks.FINGER_TAPPING_CAMERA,
            hand="right",
            trial_index=1,
        )

    assert result["algorithm_version"] == hand_tasks.HAND_ALGORITHM_VERSION
    assert result["hand_task"] == hand_tasks.FINGER_TAPPING_CAMERA
    assert result["hand"] == "right"
    assert result["trial_index"] == 1
    assert result["total_frames"] == FRAME_COUNT
    assert result["recording_quality_warnings"] == []
    assert result["tracking_confidence_summary"]["low_confidence_frames"] == 0
    assert result["tracking_confidence_summary"]["total_frames_tracked"] == FRAME_COUNT

    task_analysis = result["task_analysis"]
    assert task_analysis["task_type"] == hand_tasks.FINGER_TAPPING_CAMERA
    assert task_analysis["tap_count"] > 0
    assert task_analysis["tap_rate_hz"] == pytest.approx(2.0, abs=0.1)
    assert "wrist_stability" in task_analysis
    for key in ("tap_count", "tap_rate_hz", "mean_amplitude", "wrist_stability"):
        assert key in task_analysis["metric_units"]
        assert key in task_analysis["metric_descriptions"]


def test_run_hand_analysis_postural_tremor_end_to_end(synthetic_video):
    fake_tracker = _FakeHandTracker(_tremor_landmarks)
    with patch("hand_tasks.HandTracker", return_value=fake_tracker):
        result = hand_tasks.run_hand_analysis(
            video_path=synthetic_video,
            hand_task=hand_tasks.POSTURAL_TREMOR,
            hand="left",
        )

    assert result["hand"] == "left"
    task_analysis = result["task_analysis"]
    assert task_analysis["task_type"] == hand_tasks.POSTURAL_TREMOR
    assert task_analysis["dominant_frequency_hz"] == pytest.approx(5.0, abs=0.5)
    # 30fps on a tremor task -- below the 60fps-preferred bar -- should be a
    # soft warning, not a failure.
    assert any("60" in w for w in result["recording_quality_warnings"])


def test_run_hand_analysis_raises_recording_quality_error_when_hand_rarely_visible(synthetic_video):
    fake_tracker = _FakeHandTracker(_tapping_landmarks, visible_frame_ratio=0.2)
    with patch("hand_tasks.HandTracker", return_value=fake_tracker):
        with pytest.raises(RecordingQualityError) as exc_info:
            hand_tasks.run_hand_analysis(
                video_path=synthetic_video, hand_task=hand_tasks.FINGER_TAPPING_CAMERA,
            )

    assert any("visible" in reason for reason in exc_info.value.reasons)


def test_run_hand_analysis_surfaces_ambiguous_tracking_as_a_warning_not_a_failure(synthetic_video):
    # Fully visible, but a third of frames had only low-confidence hand
    # detection (e.g. motion blur, partial occlusion) -- should still
    # succeed, just with a warning attached, same as the pose pipeline's
    # equivalent test.
    fake_tracker = _FakeHandTracker(
        _tapping_landmarks, visible_frame_ratio=1.0, low_confidence_ratio=0.34
    )
    with patch("hand_tasks.HandTracker", return_value=fake_tracker):
        result = hand_tasks.run_hand_analysis(
            video_path=synthetic_video, hand_task=hand_tasks.FINGER_TAPPING_CAMERA,
        )

    assert any("ambiguous" in w for w in result["recording_quality_warnings"])
    assert result["tracking_confidence_summary"]["low_confidence_frames"] > 0


def test_run_hand_analysis_rejects_unknown_task(synthetic_video):
    with pytest.raises(RuntimeError):
        hand_tasks.run_hand_analysis(video_path=synthetic_video, hand_task="not_a_real_task")
