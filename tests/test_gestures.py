"""Tests for gesture_ctl.gestures — finger classification utilities."""

import pytest

from gesture_ctl.gestures import (
    finger_states,
    is_fist,
    is_open_palm,
    is_v_sign,
)
from gesture_ctl.tracker import LM

# ── Helpers ─────────────────────────────────────────────────────────────


def _base_landmarks() -> list[tuple[float, float, float]]:
    """21 landmarks with all fingers curled (tips below PIPs)."""
    lms: list[tuple[float, float, float]] = [(0.5, 0.5, 0.0)] * 21

    # For "curled": tip.y > pip.y  (lower on screen = higher y)
    # Index
    lms[LM.INDEX_PIP] = (0.5, 0.5, 0.0)
    lms[LM.INDEX_TIP] = (0.5, 0.7, 0.0)  # below PIP → curled
    # Middle
    lms[LM.MIDDLE_PIP] = (0.5, 0.5, 0.0)
    lms[LM.MIDDLE_TIP] = (0.5, 0.7, 0.0)
    # Ring
    lms[LM.RING_PIP] = (0.5, 0.5, 0.0)
    lms[LM.RING_TIP] = (0.5, 0.7, 0.0)
    # Pinky
    lms[LM.PINKY_PIP] = (0.5, 0.5, 0.0)
    lms[LM.PINKY_TIP] = (0.5, 0.7, 0.0)
    # Thumb (right hand): tip.x > ip.x → curled
    lms[LM.THUMB_IP] = (0.4, 0.5, 0.0)
    lms[LM.THUMB_TIP] = (0.5, 0.5, 0.0)  # tip to the right of IP → curled for Right

    return lms


def _extend_finger(lms: list, tip_idx: int, pip_idx: int) -> None:
    """Move tip above PIP (lower y) to extend a finger in place."""
    px, py, pz = lms[pip_idx]
    lms[tip_idx] = (px, py - 0.3, pz)


def _extend_thumb_right(lms: list) -> None:
    """Extend thumb for a right hand (tip.x < ip.x)."""
    lms[LM.THUMB_TIP] = (0.2, 0.5, 0.0)
    lms[LM.THUMB_IP] = (0.4, 0.5, 0.0)


# ── Tests ───────────────────────────────────────────────────────────────


class TestFingerStates:
    def test_all_curled_is_fist(self) -> None:
        lms = _base_landmarks()
        assert is_fist(lms, "Right")
        assert not is_open_palm(lms, "Right")
        assert not is_v_sign(lms, "Right")

    def test_all_extended_is_open_palm(self) -> None:
        lms = _base_landmarks()
        _extend_thumb_right(lms)
        _extend_finger(lms, LM.INDEX_TIP, LM.INDEX_PIP)
        _extend_finger(lms, LM.MIDDLE_TIP, LM.MIDDLE_PIP)
        _extend_finger(lms, LM.RING_TIP, LM.RING_PIP)
        _extend_finger(lms, LM.PINKY_TIP, LM.PINKY_PIP)
        assert is_open_palm(lms, "Right")
        assert not is_fist(lms, "Right")

    def test_v_sign(self) -> None:
        lms = _base_landmarks()
        _extend_finger(lms, LM.INDEX_TIP, LM.INDEX_PIP)
        _extend_finger(lms, LM.MIDDLE_TIP, LM.MIDDLE_PIP)
        # Ring and pinky stay curled
        assert is_v_sign(lms, "Right")

    def test_v_sign_fails_if_ring_extended(self) -> None:
        lms = _base_landmarks()
        _extend_finger(lms, LM.INDEX_TIP, LM.INDEX_PIP)
        _extend_finger(lms, LM.MIDDLE_TIP, LM.MIDDLE_PIP)
        _extend_finger(lms, LM.RING_TIP, LM.RING_PIP)
        assert not is_v_sign(lms, "Right")

    def test_finger_states_returns_five_bools(self) -> None:
        lms = _base_landmarks()
        states = finger_states(lms, "Right")
        assert len(states) == 5
        assert all(isinstance(s, bool) for s in states)
