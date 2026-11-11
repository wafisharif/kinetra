"""Unit tests for pose_tracking.py's pure matching/selection logic.

Deliberately uses fabricated PersonDetection objects (plain tuples for
bbox/hip_center) rather than real mediapipe output or video -- these
functions don't know or care where a detection came from, so testing them
this way is both faster and a more precise regression check than anything
involving a real model would be.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pose_tracking import (
    PersonDetection,
    hip_distance,
    iou,
    select_target,
    track_target,
)


def _det(bbox, hip_center):
    return PersonDetection(landmarks=[], bbox=bbox, hip_center=hip_center)


# --------------------------------------------------------------------------
# iou / hip_distance -- pure geometry, no detection logic involved
# --------------------------------------------------------------------------

def test_iou_identical_boxes_is_one():
    box = (0.2, 0.2, 0.6, 0.6)
    assert iou(box, box) == 1.0


def test_iou_non_overlapping_boxes_is_zero():
    assert iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0


def test_iou_partial_overlap_is_between_zero_and_one():
    value = iou((0.0, 0.0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75))
    assert 0.0 < value < 1.0


def test_hip_distance_is_euclidean():
    assert hip_distance((0.0, 0.0), (0.3, 0.4)) == 0.5  # 3-4-5 triangle


# --------------------------------------------------------------------------
# select_target -- largest-and-most-centered heuristic
# --------------------------------------------------------------------------

def test_select_target_picks_larger_box_over_smaller():
    small = _det((0.45, 0.45, 0.55, 0.55), (0.5, 0.5))  # tiny, dead center
    large = _det((0.1, 0.1, 0.9, 0.9), (0.5, 0.5))       # huge, dead center
    index = select_target([small, large])
    assert index == 1


def test_select_target_picks_centered_box_over_off_center_same_size():
    centered = _det((0.3, 0.3, 0.7, 0.7), (0.5, 0.5))
    off_center = _det((0.6, 0.6, 1.0, 1.0), (0.8, 0.8))
    index = select_target([centered, off_center])
    assert index == 0


def test_select_target_with_no_detections_returns_none():
    assert select_target([]) is None


def test_select_target_single_detection_returns_it():
    only = _det((0.3, 0.3, 0.7, 0.7), (0.5, 0.5))
    assert select_target([only]) == 0


# --------------------------------------------------------------------------
# track_target -- frame-to-frame identity tracking
# --------------------------------------------------------------------------

def test_track_target_with_no_previous_frame_falls_back_to_select_target():
    person = _det((0.3, 0.3, 0.7, 0.7), (0.5, 0.5))
    result = track_target([person], previous_hip_center=None, previous_bbox=None)
    assert result == {"target_index": 0, "confidence": "high"}


def test_track_target_with_no_detections_returns_none_confidence():
    result = track_target([], previous_hip_center=(0.5, 0.5), previous_bbox=(0.3, 0.3, 0.7, 0.7))
    assert result == {"target_index": None, "confidence": "none"}


def test_track_target_re_identifies_same_person_by_overlap():
    # Patient barely moved between frames -- high IoU with the previous box.
    previous_bbox = (0.3, 0.3, 0.7, 0.7)
    previous_hip = (0.5, 0.5)
    same_person = _det((0.31, 0.3, 0.71, 0.7), (0.51, 0.5))
    result = track_target([same_person], previous_hip, previous_bbox)
    assert result == {"target_index": 0, "confidence": "high"}


def test_track_target_does_not_swap_to_a_second_person_crossing_frame():
    # The patient (tracked) stayed roughly in place; a second person (e.g.
    # a therapist) walks through on the far side of frame. The tracker
    # should keep following the original patient, not jump to whoever is
    # now largest/most-centered (which select_target alone would do).
    previous_bbox = (0.3, 0.3, 0.7, 0.7)
    previous_hip = (0.5, 0.5)
    patient_now = _det((0.31, 0.31, 0.71, 0.71), (0.51, 0.51))
    crosser = _det((0.0, 0.0, 0.9, 0.9), (0.1, 0.1))  # huge, but far from previous hip
    result = track_target([crosser, patient_now], previous_hip, previous_bbox)
    assert result["target_index"] == 1
    assert result["confidence"] == "high"


def test_track_target_falls_back_with_low_confidence_when_nobody_matches():
    # Previous target is gone (left frame); only a new, unrelated person
    # remains. The tracker still returns someone (so Phase 3's examiner
    # tracking etc. isn't starved of landmarks) but flags low confidence.
    previous_bbox = (0.3, 0.3, 0.7, 0.7)
    previous_hip = (0.5, 0.5)
    unrelated = _det((0.0, 0.0, 0.1, 0.1), (0.05, 0.05))
    result = track_target([unrelated], previous_hip, previous_bbox)
    assert result == {"target_index": 0, "confidence": "low"}
