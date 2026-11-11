"""
pose_tracking.py -- multi-person pose detection + patient lock.

Wraps mediapipe's modern Tasks API (`mediapipe.tasks.vision.PoseLandmarker`)
instead of the legacy `mediapipe.python.solutions.pose` single-person API
that pose_test.py used before this module existed. Isolated into its own
file, with its own unit tests (tests/backend/test_pose_tracking.py), because:

1. It's the riskiest change in this effort -- swapping the actual detection
   backend -- and deserves to be tested as a pure, synthetic-data unit,
   independent of real video/mediapipe runtime behavior.
2. The reactive-balance test (a later phase, not built yet) needs more than
   one person's landmarks in the same frame (patient + examiner), which the
   legacy single-person API could never give us no matter how pose_test.py
   used it. This module returns every detected person every frame, not just
   the one chosen as the patient -- nothing is thrown away.

Everything that depends on real mediapipe landmark objects (bbox/hip-center
extraction, the PoseTracker class itself) is kept to the thin bottom section.
The matching/selection logic above it (iou, hip_distance, select_target,
track_target) operates on plain tuples and a tiny PersonDetection dataclass,
so it's testable with fabricated boxes/hip-centers -- no real video, no real
model file, no network, required.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mediapipe as mp

# Same model family/weight class the frontend's on-device WebView worker
# already downloads at runtime (see hooks/useOnDevicePose.tsx's
# MEDIAPIPE_MODEL_URL) -- "lite" trades a little accuracy for a much smaller
# download and lower memory footprint, which matters on Render's free-tier
# 512MB instance (see server.py's comment on this).
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_MODEL_CACHE_PATH = os.path.join(
    tempfile.gettempdir(), "kinetra_pose_landmarker_lite.task"
)

# Patient (up to) + examiner/bystanders. Keeping this modest (not, say, 10)
# keeps the per-frame detection cost bounded -- every extra pose MediaPipe
# tracks costs real CPU time on every single frame of the video.
MAX_POSES = 4

# Tunable in one place rather than magic numbers scattered through the
# matching logic below.
IOU_MATCH_THRESHOLD = 0.3
HIP_DISTANCE_MATCH_THRESHOLD = 0.25  # normalized (0-1) image-space distance

# BlazePose's 33-point topology -- identical under the legacy solutions API
# and the new Tasks API, so these indices are safe to hardcode here rather
# than importing the (otherwise now-unused) legacy `mp_pose.PoseLandmark`
# enum just for two constants.
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y), normalized 0-1


def ensure_model_downloaded(path: str = _MODEL_CACHE_PATH, url: str = POSE_MODEL_URL) -> str:
    """Download the .task model file once and cache it on disk. Safe to call
    on every process start -- skips the download if the file already exists
    (persists across requests within the same server process, though not
    across a fresh deploy/restart -- Render's free-tier dyno gets a clean
    filesystem on every deploy)."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    tmp_path = path + ".partial"
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, path)
    return path


@dataclass
class PersonDetection:
    """One detected person in one frame."""
    landmarks: list  # mediapipe NormalizedLandmark-like objects (x, y, z, visibility)
    bbox: BBox
    hip_center: Point
    is_target: bool = False


@dataclass
class PoseFrameResult:
    target_landmarks: Optional[list]
    all_detections: List[PersonDetection]
    tracking_confidence: str  # "high" | "low" | "none"


def _bbox_from_landmarks(landmarks) -> BBox:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


def _hip_center_from_landmarks(landmarks) -> Point:
    left_hip = landmarks[LEFT_HIP_INDEX]
    right_hip = landmarks[RIGHT_HIP_INDEX]
    return ((left_hip.x + right_hip.x) / 2.0, (left_hip.y + right_hip.y) / 2.0)


def iou(box_a: BBox, box_b: BBox) -> float:
    """Intersection-over-union of two (min_x, min_y, max_x, max_y) boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_w, inter_h = max(0.0, inter_x1 - inter_x0), max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def hip_distance(hip_a: Point, hip_b: Point) -> float:
    return ((hip_a[0] - hip_b[0]) ** 2 + (hip_a[1] - hip_b[1]) ** 2) ** 0.5


def select_target(detections: List[PersonDetection]) -> Optional[int]:
    """Pick which detected person is the patient on the FIRST frame (or any
    frame with no previous target to track from): the one whose bounding
    box is largest AND closest to frame center. Returns an index into
    `detections`, or None if there are no detections.

    Kept as its own small function (rather than inlined into track_target)
    so this specific rule can change later -- e.g. if real-world testing
    shows a different heuristic works better -- without touching the
    frame-to-frame tracker below.
    """
    if not detections:
        return None
    best_index = None
    best_score = -1.0
    for index, det in enumerate(detections):
        x0, y0, x1, y1 = det.bbox
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        box_center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        center_distance = (
            (box_center[0] - 0.5) ** 2 + (box_center[1] - 0.5) ** 2
        ) ** 0.5
        # Max possible center-distance in a 0-1 frame is sqrt(0.5) ~= 0.707;
        # normalize so "closeness" is a clean 0-1 score like area already is.
        closeness = 1.0 - min(1.0, center_distance / 0.707)
        score = area * 0.7 + closeness * 0.3
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def track_target(
    detections: List[PersonDetection],
    previous_hip_center: Optional[Point],
    previous_bbox: Optional[BBox],
) -> dict:
    """Frame-to-frame identity tracking: given this frame's detections and
    the previous frame's target hip-center/bbox, find which detection (if
    any) is the same person as before, using IoU + hip-center distance so a
    person crossing paths with the patient (e.g. a therapist walking
    through frame) doesn't cause a silent swap.

    Returns {"target_index": int | None, "confidence": "high"|"low"|"none"}.
    "low" means a match was found but only via the center/largest fallback
    (the previous target wasn't re-identified with confidence) -- callers
    should treat these frames similarly to a low-visibility frame.
    """
    if not detections:
        return {"target_index": None, "confidence": "none"}

    if previous_hip_center is None or previous_bbox is None:
        index = select_target(detections)
        return {"target_index": index, "confidence": "high" if index is not None else "none"}

    best_index = None
    best_iou = -1.0
    for index, det in enumerate(detections):
        this_iou = iou(det.bbox, previous_bbox)
        this_hip_distance = hip_distance(det.hip_center, previous_hip_center)
        if this_iou >= IOU_MATCH_THRESHOLD or this_hip_distance <= HIP_DISTANCE_MATCH_THRESHOLD:
            if this_iou > best_iou:
                best_iou = this_iou
                best_index = index

    if best_index is None:
        # Nobody matched well enough -- most likely the target left frame or
        # was briefly occluded. Fall back to the center/largest heuristic
        # rather than silently losing the patient for the rest of the video.
        index = select_target(detections)
        return {"target_index": index, "confidence": "low" if index is not None else "none"}

    confidence = "high" if best_iou >= IOU_MATCH_THRESHOLD else "low"
    return {"target_index": best_index, "confidence": confidence}


class PoseTracker:
    """Stateful wrapper around mediapipe's PoseLandmarker (Tasks API, VIDEO
    mode), replacing pose_test.py's previous direct `mp_pose.Pose()` use.

    Usage mirrors the old `mp_pose.Pose()` + `.process(frame)` shape closely
    enough that pose_test.py's call site stays a small, readable diff:

        tracker = PoseTracker()
        for frame_index, frame_rgb in enumerate(frames):
            timestamp_ms = int(frame_index / fps * 1000)
            result = tracker.process(frame_rgb, timestamp_ms)
            target_landmarks = result.target_landmarks  # or None
            everyone = result.all_detections  # all people, for later phases
        tracker.close()
    """

    def __init__(self, num_poses: int = MAX_POSES, model_path: Optional[str] = None):
        resolved_path = model_path or ensure_model_downloaded()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=resolved_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=num_poses,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._previous_hip_center: Optional[Point] = None
        self._previous_bbox: Optional[BBox] = None
        # Running totals the caller (pose_test.py) folds into its
        # recording_quality.py check after the frame loop finishes.
        self.low_confidence_frame_count = 0
        self.total_frame_count = 0

    def process(self, frame_rgb, timestamp_ms: int) -> PoseFrameResult:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        self.total_frame_count += 1
        detections: List[PersonDetection] = []
        for landmarks in (result.pose_landmarks or []):
            detections.append(PersonDetection(
                landmarks=landmarks,
                bbox=_bbox_from_landmarks(landmarks),
                hip_center=_hip_center_from_landmarks(landmarks),
            ))

        tracked = track_target(detections, self._previous_hip_center, self._previous_bbox)
        target_index = tracked["target_index"]

        if tracked["confidence"] == "low":
            self.low_confidence_frame_count += 1

        target_landmarks = None
        if target_index is not None:
            detections[target_index].is_target = True
            target_landmarks = detections[target_index].landmarks
            self._previous_hip_center = detections[target_index].hip_center
            self._previous_bbox = detections[target_index].bbox

        return PoseFrameResult(
            target_landmarks=target_landmarks,
            all_detections=detections,
            tracking_confidence=tracked["confidence"],
        )

    def close(self):
        self._landmarker.close()
