"""Tests for gesture_ctl.state_machine — Pinch FSM transitions."""

import pytest

from gesture_ctl.config import Config
from gesture_ctl.state_machine import GestureEvent, PinchStateMachine
from gesture_ctl.tracker import LM

# ── Helpers ─────────────────────────────────────────────────────────────

FAR = 0.15   # well above threshold
CLOSE = 0.02  # well below threshold


def _make_landmarks(
    thumb_index_dist: float = FAR,
    thumb_middle_dist: float = FAR,
) -> list[tuple[float, float, float]]:
    """Build a minimal 21-landmark list with controllable pinch distances.

    Only landmarks 4 (thumb tip), 8 (index tip), and 12 (middle tip) matter
    for the FSM; the rest are filled with zeros.
    """
    lms: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 21
    # Thumb tip at origin
    lms[LM.THUMB_TIP] = (0.5, 0.5, 0.0)
    # Index tip at controlled distance
    lms[LM.INDEX_TIP] = (0.5 + thumb_index_dist, 0.5, 0.0)
    # Middle tip at controlled distance
    lms[LM.MIDDLE_TIP] = (0.5 + thumb_middle_dist, 0.5, 0.0)
    return lms


@pytest.fixture
def cfg() -> Config:
    return Config(
        pinch_threshold=0.045,
        quick_pinch_ms=300,
        double_pinch_window_ms=400,
        drag_hold_ms=300,
    )


@pytest.fixture
def fsm(cfg: Config) -> PinchStateMachine:
    return PinchStateMachine(cfg)


# ── Left click tests ───────────────────────────────────────────────────

class TestLeftClick:
    def test_quick_pinch_single_click(self, fsm: PinchStateMachine) -> None:
        """Pinch < 300ms then release → single click (after timeout)."""
        t = 0.0
        # Frame 1: pinch
        events = fsm.update(_make_landmarks(CLOSE), t)
        assert events == []

        # Frame 2: release after 100ms
        t += 0.1
        events = fsm.update(_make_landmarks(FAR), t)
        assert events == []  # enters CLICK_CANDIDATE, waiting for double

        # Frame 3: timeout (> 400ms window)
        t += 0.5
        events = fsm.update(_make_landmarks(FAR), t)
        assert GestureEvent.LEFT_CLICK in events

    def test_double_click(self, fsm: PinchStateMachine) -> None:
        """Two quick pinches within 400ms → double click."""
        t = 0.0

        # First pinch
        fsm.update(_make_landmarks(CLOSE), t)
        t += 0.1
        fsm.update(_make_landmarks(FAR), t)  # release → CLICK_CANDIDATE

        # Second pinch within 400ms
        t += 0.15
        fsm.update(_make_landmarks(CLOSE), t)  # re-pinch
        t += 0.05
        events = fsm.update(_make_landmarks(FAR), t)  # release
        # The double-click fires on the release of the second pinch
        # (via PINCHED→release path detecting last_click within dbl window)
        assert GestureEvent.LEFT_DOUBLE_CLICK in events

    def test_drag_start_and_end(self, fsm: PinchStateMachine) -> None:
        """Hold pinch ≥ 300ms → drag start; release → drag end."""
        t = 0.0

        # Pinch and hold
        fsm.update(_make_landmarks(CLOSE), t)
        t += 0.35  # 350ms > drag_hold_ms
        events = fsm.update(_make_landmarks(CLOSE), t)
        assert GestureEvent.LEFT_DRAG_START in events

        # Continue holding — no new events
        t += 0.1
        events = fsm.update(_make_landmarks(CLOSE), t)
        assert events == []

        # Release
        t += 0.1
        events = fsm.update(_make_landmarks(FAR), t)
        assert GestureEvent.LEFT_DRAG_END in events


# ── Right click tests ──────────────────────────────────────────────────

class TestRightClick:
    def test_quick_pinch_right_click(self, fsm: PinchStateMachine) -> None:
        """Thumb + middle quick pinch → right click."""
        t = 0.0
        fsm.update(_make_landmarks(thumb_middle_dist=CLOSE), t)
        t += 0.1
        fsm.update(_make_landmarks(thumb_middle_dist=FAR), t)
        t += 0.5
        events = fsm.update(_make_landmarks(thumb_middle_dist=FAR), t)
        assert GestureEvent.RIGHT_CLICK in events

    def test_right_drag(self, fsm: PinchStateMachine) -> None:
        """Hold thumb + middle ≥ 300ms → right drag start."""
        t = 0.0
        fsm.update(_make_landmarks(thumb_middle_dist=CLOSE), t)
        t += 0.35
        events = fsm.update(_make_landmarks(thumb_middle_dist=CLOSE), t)
        assert GestureEvent.RIGHT_DRAG_START in events

        t += 0.1
        events = fsm.update(_make_landmarks(thumb_middle_dist=FAR), t)
        assert GestureEvent.RIGHT_DRAG_END in events


# ── Edge cases ──────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_events_when_idle(self, fsm: PinchStateMachine) -> None:
        """No pinch → no events."""
        for _ in range(10):
            events = fsm.update(_make_landmarks(FAR, FAR), 0.0)
            assert events == []

    def test_simultaneous_left_and_right(self, fsm: PinchStateMachine) -> None:
        """Both pinches at once should fire both FSMs independently."""
        t = 0.0
        fsm.update(_make_landmarks(CLOSE, CLOSE), t)
        t += 0.35
        events = fsm.update(_make_landmarks(CLOSE, CLOSE), t)
        # Both should fire drag starts
        assert GestureEvent.LEFT_DRAG_START in events
        assert GestureEvent.RIGHT_DRAG_START in events
