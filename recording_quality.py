"""
recording_quality.py -- pre-flight checks on a recording's aggregate
per-frame signal, so a bad recording fails with a clear, actionable reason
instead of silently producing a low-confidence (or simply wrong) score.

Deliberately pure and video-agnostic: every function here takes plain
numbers (frame counts, fps, a visibility ratio) rather than raw frames or
mediapipe objects, so it's fully unit-testable with synthetic data (see
tests/backend/test_recording_quality.py) and reusable by both the body-task
pipeline (pose_test.py) and the hand-task pipeline (hand_tasks.py, Phase 1)
without either depending on the other or on mediapipe/opencv being
importable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# Required-landmarks visibility, across the whole recording, before a
# result is trustworthy enough to show the person. 90% per the brief.
MIN_VISIBLE_FRAME_RATIO = 0.90

# Below this, the recording is still analyzed -- just with a warning, not a
# hard failure -- since fps mostly affects precision (rate/frequency
# metrics), not whether the analysis is possible at all.
MIN_FPS = 30.0

# A near-empty/instant recording is never usable regardless of visibility.
MIN_DURATION_SECONDS = 1.0

# Above this fraction of frames with only low-confidence patient tracking
# (see pose_tracking.py's track_target), warn that multiple people in frame
# may have made tracking ambiguous.
LOW_TRACKING_CONFIDENCE_WARN_RATIO = 0.15


class RecordingQualityError(Exception):
    """Raised when a recording fails a HARD quality requirement (not visible
    enough, not long enough, nobody detected at all). Carries the specific
    reasons so the caller (server.py's /analyze and /analyze-hand routes)
    can return them to the app verbatim as a 422, rather than a generic
    message or -- worse -- a confidently wrong score."""

    def __init__(self, reasons: List[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass
class QualityCheckResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)    # hard-failure reasons; empty if passed
    warnings: List[str] = field(default_factory=list)   # soft issues; recording still usable
    visible_frame_ratio: float = 0.0
    low_confidence_tracking_ratio: float = 0.0


def check_recording_quality(
    total_frames: int,
    visible_frame_count: int,
    fps: float,
    low_confidence_tracking_count: int = 0,
) -> QualityCheckResult:
    """Run every quality rule against one recording's aggregate per-frame
    stats, collected during the (unavoidable) per-frame detection pass --
    this never re-reads the video itself, it only summarizes counts the
    caller already has after that pass finishes.

    Hard failures (`reasons`, `passed=False`):
      - no frames could be read at all
      - recording shorter than MIN_DURATION_SECONDS
      - fewer than MIN_VISIBLE_FRAME_RATIO of frames had the tracked
        person's landmarks visible (nobody detected, or lost for most of it)

    Soft warnings (recording still analyzed):
      - fps below MIN_FPS
      - a meaningful fraction of frames had only low-confidence patient
        tracking (likely more than one person in frame)
    """
    reasons: List[str] = []
    warnings: List[str] = []

    duration_seconds = (total_frames / fps) if fps else 0.0

    if total_frames <= 0:
        reasons.append("No frames could be read from this recording.")
    elif duration_seconds < MIN_DURATION_SECONDS:
        reasons.append(
            f"Recording is too short ({duration_seconds:.1f}s) to analyze. "
            f"Record at least {MIN_DURATION_SECONDS:.0f} second(s)."
        )

    visible_frame_ratio = (visible_frame_count / total_frames) if total_frames else 0.0
    if total_frames > 0 and visible_frame_ratio < MIN_VISIBLE_FRAME_RATIO:
        reasons.append(
            "The person wasn't clearly visible for enough of this recording "
            f"({visible_frame_ratio * 100:.0f}% of frames, need "
            f"{MIN_VISIBLE_FRAME_RATIO * 100:.0f}%). Re-record with the full "
            "body (or hand, for a hand check) in frame and good lighting."
        )

    if fps and fps < MIN_FPS:
        warnings.append(
            f"This recording's frame rate ({fps:.0f}fps) is below the "
            f"recommended {MIN_FPS:.0f}fps minimum. Results may be less precise."
        )

    low_confidence_tracking_ratio = (
        (low_confidence_tracking_count / total_frames) if total_frames else 0.0
    )
    if total_frames > 0 and low_confidence_tracking_ratio > LOW_TRACKING_CONFIDENCE_WARN_RATIO:
        warnings.append(
            "Tracking was ambiguous (likely multiple people in frame) for "
            f"{low_confidence_tracking_ratio * 100:.0f}% of this recording. "
            "For best results, keep only the person being tested in frame."
        )

    return QualityCheckResult(
        passed=len(reasons) == 0,
        reasons=reasons,
        warnings=warnings,
        visible_frame_ratio=visible_frame_ratio,
        low_confidence_tracking_ratio=low_confidence_tracking_ratio,
    )
