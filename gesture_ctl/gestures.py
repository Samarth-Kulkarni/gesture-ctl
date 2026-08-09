"""Phase 4 — Finger-state classification and system gesture detection.

Utility functions that inspect the 21 MediaPipe hand landmarks to
recognise higher-level poses: V-sign, open palm, fist, and per-finger
extension state.

Also provides a ``StationaryDetector`` that checks whether key landmarks
have barely moved over a configurable number of frames — used by the
toggle and play/pause gestures.
"""

from __future__ import annotations

import math
from collections import deque

from gesture_ctl.config import Config
from gesture_ctl.tracker import LM


# ── Finger indices (tip, pip) for extension check ───────────────────────

_FINGER_TIPS = [LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP]
_FINGER_PIPS = [LM.INDEX_PIP, LM.MIDDLE_PIP, LM.RING_PIP, LM.PINKY_PIP]


# ── Per-finger helpers ──────────────────────────────────────────────────

def is_finger_extended(
    landmarks: list[tuple[float, float, float]],
    finger_tip: int,
    finger_pip: int,
) -> bool:
    """Return True if the finger tip is above (lower y) its PIP joint.

    Works for index, middle, ring, pinky.  Thumb uses a separate check.
    """
    return landmarks[finger_tip][1] < landmarks[finger_pip][1]


def is_thumb_extended(
    landmarks: list[tuple[float, float, float]],
    handedness: str = "Right",
) -> bool:
    """Return True if the thumb is extended outward.

    For a right hand (camera-mirrored), thumb tip x < thumb IP x means
    extended.  For left hand it's the opposite.
    """
    tip_x = landmarks[LM.THUMB_TIP][0]
    ip_x = landmarks[LM.THUMB_IP][0]
    if handedness == "Right":
        return tip_x < ip_x
    return tip_x > ip_x


# ── High-level poses ───────────────────────────────────────────────────

def finger_states(
    landmarks: list[tuple[float, float, float]],
    handedness: str = "Right",
) -> list[bool]:
    """Return a 5-element list ``[thumb, index, middle, ring, pinky]``
    where ``True`` means extended."""
    states = [is_thumb_extended(landmarks, handedness)]
    for tip, pip_ in zip(_FINGER_TIPS, _FINGER_PIPS):
        states.append(is_finger_extended(landmarks, tip, pip_))
    return states


def is_v_sign(
    landmarks: list[tuple[float, float, float]],
    handedness: str = "Right",
) -> bool:
    """Index and middle extended, ring and pinky curled."""
    s = finger_states(landmarks, handedness)
    # s = [thumb, index, middle, ring, pinky]
    return s[1] and s[2] and not s[3] and not s[4]


def is_open_palm(
    landmarks: list[tuple[float, float, float]],
    handedness: str = "Right",
) -> bool:
    """All five fingers extended."""
    return all(finger_states(landmarks, handedness))


def is_fist(
    landmarks: list[tuple[float, float, float]],
    handedness: str = "Right",
) -> bool:
    """All five fingers curled."""
    return not any(finger_states(landmarks, handedness))


# ── Stationary detector ────────────────────────────────────────────────

class StationaryDetector:
    """Track whether key landmarks stay within a small radius over N frames.

    Used for "hold this pose for 1 second" style gestures.
    """

    def __init__(self, cfg: Config, tracked_indices: list[int] | None = None) -> None:
        self._threshold = cfg.stationary_threshold
        self._history: deque[list[tuple[float, float, float]]] = deque(maxlen=60)
        self._indices = tracked_indices or [LM.WRIST, LM.INDEX_TIP, LM.MIDDLE_TIP]

    def push(self, landmarks: list[tuple[float, float, float]]) -> None:
        """Record one frame of landmark data."""
        self._history.append(landmarks)

    def is_stationary(self, n_frames: int) -> bool:
        """Return True if tracked landmarks moved less than threshold
        over the last *n_frames* frames."""
        if len(self._history) < n_frames:
            return False

        recent = list(self._history)[-n_frames:]
        ref = recent[0]

        for frame_lms in recent[1:]:
            for idx in self._indices:
                dx = frame_lms[idx][0] - ref[idx][0]
                dy = frame_lms[idx][1] - ref[idx][1]
                dz = frame_lms[idx][2] - ref[idx][2]
                if math.sqrt(dx * dx + dy * dy + dz * dz) > self._threshold:
                    return False
        return True

    def clear(self) -> None:
        self._history.clear()
