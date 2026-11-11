"""Unit tests for recording_quality.py. Every input here is a plain number
-- no video, no mediapipe, no network -- since check_recording_quality()
only ever summarizes counts its caller already computed during the
per-frame detection pass."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from recording_quality import (
    MIN_FPS,
    MIN_VISIBLE_FRAME_RATIO,
    check_recording_quality,
)


def test_good_recording_passes_with_no_reasons_or_warnings():
    result = check_recording_quality(
        total_frames=300, visible_frame_count=300, fps=30.0,
    )
    assert result.passed is True
    assert result.reasons == []
    assert result.warnings == []
    assert result.visible_frame_ratio == 1.0


def test_no_frames_at_all_is_a_hard_failure():
    result = check_recording_quality(total_frames=0, visible_frame_count=0, fps=30.0)
    assert result.passed is False
    assert any("No frames" in reason for reason in result.reasons)


def test_too_short_recording_is_a_hard_failure():
    # 10 frames at 30fps is 0.33s -- under the 1-second floor.
    result = check_recording_quality(total_frames=10, visible_frame_count=10, fps=30.0)
    assert result.passed is False
    assert any("too short" in reason for reason in result.reasons)


def test_low_visibility_is_a_hard_failure():
    # Only 50% of frames had the person visible -- well under the 90% bar.
    result = check_recording_quality(total_frames=300, visible_frame_count=150, fps=30.0)
    assert result.passed is False
    assert any("wasn't clearly visible" in reason for reason in result.reasons)
    assert result.visible_frame_ratio == 0.5


def test_visibility_right_at_the_threshold_passes():
    total = 300
    visible = int(total * MIN_VISIBLE_FRAME_RATIO)  # exactly 90%
    result = check_recording_quality(total_frames=total, visible_frame_count=visible, fps=30.0)
    assert result.passed is True


def test_visibility_just_under_the_threshold_fails():
    total = 300
    visible = int(total * MIN_VISIBLE_FRAME_RATIO) - 1
    result = check_recording_quality(total_frames=total, visible_frame_count=visible, fps=30.0)
    assert result.passed is False


def test_low_fps_is_a_warning_not_a_hard_failure():
    result = check_recording_quality(total_frames=300, visible_frame_count=300, fps=20.0)
    assert result.passed is True  # still analyzed
    assert result.reasons == []
    assert any(f"{MIN_FPS:.0f}fps" in w for w in result.warnings)


def test_ambiguous_tracking_is_a_warning_not_a_hard_failure():
    result = check_recording_quality(
        total_frames=300, visible_frame_count=300, fps=30.0,
        low_confidence_tracking_count=100,  # 33% ambiguous
    )
    assert result.passed is True
    assert any("ambiguous" in w for w in result.warnings)


def test_ambiguous_tracking_below_warn_ratio_produces_no_warning():
    result = check_recording_quality(
        total_frames=300, visible_frame_count=300, fps=30.0,
        low_confidence_tracking_count=10,  # ~3%, well under the 15% warn bar
    )
    assert result.warnings == []


def test_multiple_hard_failures_are_all_reported_together():
    result = check_recording_quality(total_frames=5, visible_frame_count=1, fps=30.0)
    assert result.passed is False
    assert len(result.reasons) >= 2  # too short AND not visible enough
