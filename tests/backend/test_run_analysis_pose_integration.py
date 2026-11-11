"""Integration-level tests for pose_test.run_analysis()'s NEW wiring: the
Phase 0 patient-lock/quality-check plumbing added on top of the existing
analysis pipeline.

test_pose_tracking.py and test_recording_quality.py already cover those two
modules' own logic in isolation with fabricated data. What they can't catch
is a wiring mistake in pose_test.py itself -- e.g. passing the wrong value
into check_recording_quality(), or landmarks being read with the wrong
shape after the Tasks-API swap. This file exercises run_analysis() as a
whole, with a real (tiny, synthetic) video file, but a stand-in PoseTracker
so no real mediapipe model download/inference is needed -- that stays out
of scope for a fast, network-free unit test; see the plan doc for why real
on-device verification is still pending a real .task model file.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pose_test
from recording_quality import RecordingQualityError

FRAME_COUNT = 60
FPS = 30.0


class _FakeLandmark:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _make_standing_person_landmarks():
    """33 landmarks shaped roughly like someone standing facing the camera
    -- not biomechanically precise, just plausible enough (hips below
    shoulders, feet below knees, etc.) that every index pose_test.py reads
    for any task exists and produces a finite angle rather than a crash."""
    # Defaults: center-of-frame, mid-height.
    landmarks = [_FakeLandmark(0.5, 0.5) for _ in range(33)]
    positions = {
        11: (0.4, 0.3),   # LEFT_SHOULDER
        12: (0.6, 0.3),   # RIGHT_SHOULDER
        13: (0.35, 0.45),  # LEFT_ELBOW
        14: (0.65, 0.45),  # RIGHT_ELBOW
        15: (0.3, 0.6),   # LEFT_WRIST
        16: (0.7, 0.6),   # RIGHT_WRIST
        23: (0.42, 0.6),  # LEFT_HIP
        24: (0.58, 0.6),  # RIGHT_HIP
        25: (0.42, 0.8),  # LEFT_KNEE
        26: (0.58, 0.8),  # RIGHT_KNEE
        27: (0.42, 0.95),  # LEFT_ANKLE
        28: (0.58, 0.95),  # RIGHT_ANKLE
        29: (0.41, 0.97),  # LEFT_HEEL
        30: (0.59, 0.97),  # RIGHT_HEEL
        31: (0.43, 0.98),  # LEFT_FOOT_INDEX
        32: (0.57, 0.98),  # RIGHT_FOOT_INDEX
    }
    for index, (x, y) in positions.items():
        landmarks[index] = _FakeLandmark(x, y)
    return landmarks


class _FakePoseFrameResult:
    def __init__(self, target_landmarks, tracking_confidence="high"):
        self.target_landmarks = target_landmarks
        self.all_detections = []
        self.tracking_confidence = tracking_confidence


class _FakePoseTracker:
    """Stand-in for pose_tracking.PoseTracker: same shape (process/close +
    the two running counters run_analysis reads), but returns canned,
    controllable results per frame instead of running real inference."""

    def __init__(self, visible_frame_ratio=1.0, low_confidence_ratio=0.0):
        self.visible_frame_ratio = visible_frame_ratio
        self.low_confidence_ratio = low_confidence_ratio
        self.low_confidence_frame_count = 0
        self.total_frame_count = 0
        self._landmarks = _make_standing_person_landmarks()

    def process(self, frame_rgb, timestamp_ms):
        self.total_frame_count += 1
        is_visible = (self.total_frame_count % 100) < (self.visible_frame_ratio * 100)
        is_low_confidence = (self.total_frame_count % 100) < (self.low_confidence_ratio * 100)
        if is_low_confidence:
            self.low_confidence_frame_count += 1
        landmarks = self._landmarks if is_visible else None
        confidence = "low" if is_low_confidence else "high"
        return _FakePoseFrameResult(landmarks, confidence)

    def close(self):
        pass


def _write_synthetic_video(path: str, frame_count: int = FRAME_COUNT, fps: float = FPS):
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64)
    )
    for i in range(frame_count):
        # Slightly varying solid-color frames -- content doesn't matter
        # since detection is faked; just needs to be a decodable video.
        frame = np.full((64, 64, 3), fill_value=(i % 255), dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_video(tmp_path):
    video_path = str(tmp_path / "synthetic.mp4")
    _write_synthetic_video(video_path)
    return video_path


def test_run_analysis_good_recording_includes_new_phase0_fields(synthetic_video, tmp_path):
    fake_tracker = _FakePoseTracker(visible_frame_ratio=1.0, low_confidence_ratio=0.0)
    with patch("pose_test.PoseTracker", return_value=fake_tracker):
        result = pose_test.run_analysis(
            video_path=synthetic_video,
            mode="daily",
            daily_task="reach",
            output_csv=str(tmp_path / "angles.csv"),
            backend_mode=True,
            trial_index=2,
        )

    assert result["algorithm_version"] == pose_test.ALGORITHM_VERSION
    assert result["trial_index"] == 2
    assert result["recording_quality_warnings"] == []
    assert result["tracking_confidence_summary"]["low_confidence_frames"] == 0
    assert result["tracking_confidence_summary"]["total_frames_tracked"] == FRAME_COUNT
    assert result["total_frames"] == FRAME_COUNT


def test_run_analysis_raises_recording_quality_error_when_person_rarely_visible(
    synthetic_video, tmp_path
):
    # Visible in only ~20% of frames -- well under the 90% bar -- should
    # raise before any task summarization runs, not return a misleading
    # low-confidence score.
    fake_tracker = _FakePoseTracker(visible_frame_ratio=0.2)
    with patch("pose_test.PoseTracker", return_value=fake_tracker):
        with pytest.raises(RecordingQualityError) as exc_info:
            pose_test.run_analysis(
                video_path=synthetic_video,
                mode="daily",
                daily_task="reach",
                output_csv=str(tmp_path / "angles.csv"),
                backend_mode=True,
            )

    assert any("visible" in reason for reason in exc_info.value.reasons)


def test_run_analysis_surfaces_ambiguous_tracking_as_a_warning_not_a_failure(
    synthetic_video, tmp_path
):
    # Fully visible, but a third of frames had only low-confidence patient
    # tracking (e.g. a second person crossed frame) -- should still
    # succeed, just with a warning attached.
    fake_tracker = _FakePoseTracker(visible_frame_ratio=1.0, low_confidence_ratio=0.34)
    with patch("pose_test.PoseTracker", return_value=fake_tracker):
        result = pose_test.run_analysis(
            video_path=synthetic_video,
            mode="daily",
            daily_task="reach",
            output_csv=str(tmp_path / "angles.csv"),
            backend_mode=True,
        )

    assert any("ambiguous" in w for w in result["recording_quality_warnings"])
    assert result["tracking_confidence_summary"]["low_confidence_frames"] > 0
