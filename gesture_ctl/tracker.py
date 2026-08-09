"""Phase 1 — MediaPipe Hands landmark extraction (Tasks API).

Uses ``mediapipe.tasks.python.vision.HandLandmarker`` (the current API
in MediaPipe ≥ 1.0).  The legacy ``mp.solutions.hands`` has been removed.

Requires the ``hand_landmarker.task`` model file — see :func:`ensure_model`
for auto-download logic.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger(__name__)

# ── Model path management ──────────────────────────────────────────────

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_MODEL_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODEL_DIR / "hand_landmarker.task"


def ensure_model() -> str:
    """Download the hand-landmarker model if it doesn't exist yet.

    Returns the absolute path to the ``.task`` file.
    """
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)

    logger.info("Downloading hand_landmarker.task model …")
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    logger.info("Model saved to %s", _MODEL_PATH)
    return str(_MODEL_PATH)


# ── Landmark indices (for readability elsewhere) ────────────────────────

class LM:
    """MediaPipe hand landmark indices."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# ── Hand connection pairs for drawing ──────────────────────────────────

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle  (wrist→MCP via 0→9)
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


@dataclass
class HandResult:
    """Processed hand-tracking output for a single frame."""

    detected: bool = False
    landmarks: list[tuple[float, float, float]] = field(default_factory=list)
    handedness: str = ""  # "Left" or "Right"


class HandTracker:
    """Real-time hand tracker using MediaPipe Tasks HandLandmarker.

    Parameters
    ----------
    max_hands : int
        Track at most this many hands (default 1 for air-mouse).
    detection_confidence : float
        Minimum confidence for the initial detection.
    tracking_confidence : float
        Minimum confidence for frame-to-frame tracking.
    """

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
    ) -> None:
        model_path = ensure_model()

        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_ts_ms: int = 0

    # ── Core processing ─────────────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray) -> HandResult:
        """Run hand detection on a BGR frame.

        Returns a :class:`HandResult` with normalised landmarks.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp must be monotonically increasing
        self._frame_ts_ms += 1
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        if not result.hand_landmarks:
            return HandResult(detected=False)

        hand_lms = result.hand_landmarks[0]
        handedness_label = (
            result.handedness[0][0].category_name
            if result.handedness
            else ""
        )

        landmarks = [
            (lm.x, lm.y, lm.z) for lm in hand_lms
        ]

        return HandResult(
            detected=True,
            landmarks=landmarks,
            handedness=handedness_label,
        )

    # ── Debug drawing (manual, since mp.solutions.drawing_utils is gone) ─

    def draw_landmarks(
        self,
        bgr_frame: np.ndarray,
        result: HandResult,
        colour: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        radius: int = 4,
    ) -> np.ndarray:
        """Draw hand skeleton onto the frame (in-place) for the debug preview."""
        if not result.detected:
            return bgr_frame

        h, w = bgr_frame.shape[:2]
        pts = [(int(x * w), int(y * h)) for x, y, _z in result.landmarks]

        # Draw connections
        for a, b in HAND_CONNECTIONS:
            cv2.line(bgr_frame, pts[a], pts[b], colour, thickness)

        # Draw landmark dots
        for px, py in pts:
            cv2.circle(bgr_frame, (px, py), radius, (255, 255, 255), -1)
            cv2.circle(bgr_frame, (px, py), radius - 1, colour, -1)

        return bgr_frame

    # ── Cleanup ─────────────────────────────────────────────────────────

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
