"""Tests for gesture_ctl.state_machine — Pinch FSM transitions."""

import pytest

from gesture_ctl.config import Config
from gesture_ctl.state_machine import GestureEvent, PinchStateMachine
from gesture_ctl.tracker import LM

# ── Helpers ─────────────────────────────────────────────────────────────

FAR = 0.15   # well above threshold
CLOSE = 0.02  # well below threshold
MID = 0.050  # between pinch_threshold (0.040) and release_threshold (0.060)


def _make_landmarks(
    thumb_index_dist: float = FAR,
    thumb_middle_dist: float = FAR,
    thumb_ring_dist: float = FAR,
    thumb_pinky_dist: float = FAR,
) -> list[tuple[float, float, float]]:
    """Build a minimal 21-landmark list with controllable pinch distances.

    Only landmarks 4 (thumb tip), 8 (index tip), 12 (middle tip),
    16 (ring tip), and 20 (pinky tip) matter for the FSM; the rest
    are filled with zeros.
    """
    lms: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 21
    # Thumb tip at origin
    lms[LM.THUMB_TIP] = (0.5, 0.5, 0.0)
    # Index tip at controlled distance
    lms[LM.INDEX_TIP] = (0.5 + thumb_index_dist, 0.5, 0.0)
    # Middle tip at controlled distance
    lms[LM.MIDDLE_TIP] = (0.5 + thumb_middle_dist, 0.5, 0.0)
    # Ring tip at controlled distance
    lms[LM.RING_TIP] = (0.5 + thumb_ring_dist, 0.5, 0.0)
    # Pinky tip at controlled distance
    lms[LM.PINKY_TIP] = (0.5 + thumb_pinky_dist, 0.5, 0.0)
    return lms


@pytest.fixture
def cfg() -> Config:
    return Config(
        pinch_threshold=0.045,
        pinch_release_threshold=0.060,
        drag_release_frames=2,
        quick_pinch_ms=300,
        double_pinch_window_ms=400,
        drag_hold_ms=300,
        scroll_threshold=0.045,
        scroll_release_threshold=0.060,
        scroll_interval_ms=50.0,
        scroll_step=120,
    )


@pytest.fixture
def fsm(cfg: Config) -> PinchStateMachine:
    return PinchStateMachine(cfg)


# ── Left click tests ───────────────────────────────────────────────────

class TestLeftClick:
    def test_quick_pinch_single_click(self, fsm: PinchStateMachine) -> None:
        """Pinch < 300ms then release → instant single click on release."""
        t = 0.0
        # Frame 1: pinch
        events = fsm.update(_make_landmarks(CLOSE), t)
        assert events == []

        # Frame 2: release after 100ms → instant single click!
        t += 0.1
        events = fsm.update(_make_landmarks(FAR), t)
        assert GestureEvent.LEFT_CLICK in events

    def test_double_click(self, fsm: PinchStateMachine) -> None:
        """Two quick pinches within 400ms → double click on 2nd release."""
        t = 1.0  # start at 1.0s to ensure clear last_click state

        # First pinch release
        fsm.update(_make_landmarks(CLOSE), t)
        t += 0.1
        fsm.update(_make_landmarks(FAR), t)  # instant 1st click

        # Second pinch release within 350ms
        t += 0.15
        fsm.update(_make_landmarks(CLOSE), t)  # re-pinch
        t += 0.05
        events = fsm.update(_make_landmarks(FAR), t)  # 2nd release
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

        # Release — need 2 consecutive frames above release threshold
        t += 0.1
        events = fsm.update(_make_landmarks(FAR), t)
        assert events == []  # first frame above threshold, not enough

        t += 0.033
        events = fsm.update(_make_landmarks(FAR), t)
        assert GestureEvent.LEFT_DRAG_END in events


# ── Right click tests ──────────────────────────────────────────────────

class TestRightClick:
    def test_quick_pinch_right_click(self, fsm: PinchStateMachine) -> None:
        """Thumb + middle quick pinch → instant right click on release."""
        t = 1.0
        fsm.update(_make_landmarks(thumb_middle_dist=CLOSE), t)
        t += 0.1
        events = fsm.update(_make_landmarks(thumb_middle_dist=FAR), t)
        assert GestureEvent.RIGHT_CLICK in events

    def test_right_drag(self, fsm: PinchStateMachine) -> None:
        """Hold thumb + middle ≥ 300ms → right drag start."""
        t = 0.0
        fsm.update(_make_landmarks(thumb_middle_dist=CLOSE), t)
        t += 0.35
        events = fsm.update(_make_landmarks(thumb_middle_dist=CLOSE), t)
        assert GestureEvent.RIGHT_DRAG_START in events

        # Release with 2 consecutive frames
        t += 0.1
        fsm.update(_make_landmarks(thumb_middle_dist=FAR), t)
        t += 0.033
        events = fsm.update(_make_landmarks(thumb_middle_dist=FAR), t)
        assert GestureEvent.RIGHT_DRAG_END in events


# ── Hysteresis tests ───────────────────────────────────────────────────

class TestHysteresis:
    def test_drag_hysteresis_prevents_accidental_release(self, fsm: PinchStateMachine) -> None:
        """Distance fluctuating between threshold and release_threshold
        should NOT drop the drag."""
        t = 0.0

        # Start drag
        fsm.update(_make_landmarks(CLOSE), t)
        t += 0.35
        events = fsm.update(_make_landmarks(CLOSE), t)
        assert GestureEvent.LEFT_DRAG_START in events

        # Fluctuate in the hysteresis band (0.040–0.060) — should stay in DRAGGING
        for _ in range(10):
            t += 0.033
            events = fsm.update(_make_landmarks(MID), t)
            assert events == [], "Mid-band distance should not drop drag"

    def test_drag_requires_consecutive_release_frames(self, fsm: PinchStateMachine) -> None:
        """A single frame above release threshold should NOT end drag."""
        t = 0.0

        # Start drag
        fsm.update(_make_landmarks(CLOSE), t)
        t += 0.35
        events = fsm.update(_make_landmarks(CLOSE), t)
        assert GestureEvent.LEFT_DRAG_START in events

        # One frame above release threshold
        t += 0.033
        events = fsm.update(_make_landmarks(FAR), t)
        assert events == [], "Single frame should not end drag"

        # Back below release threshold — resets counter
        t += 0.033
        events = fsm.update(_make_landmarks(CLOSE), t)
        assert events == [], "Should still be dragging"

        # One more frame above — again only 1 consecutive
        t += 0.033
        events = fsm.update(_make_landmarks(FAR), t)
        assert events == [], "Still only 1 consecutive frame"


# ── Scroll tests ───────────────────────────────────────────────────────

class TestScroll:
    def test_scroll_down_ring_thumb(self, fsm: PinchStateMachine) -> None:
        """Pinching ring+thumb should produce SCROLL_DOWN events."""
        t = 0.0

        # First pinch → immediate scroll event
        events = fsm.update(_make_landmarks(thumb_ring_dist=CLOSE), t)
        assert GestureEvent.SCROLL_DOWN in events

        # Holding for 60ms (> 50ms interval) → another scroll
        t += 0.060
        events = fsm.update(_make_landmarks(thumb_ring_dist=CLOSE), t)
        assert GestureEvent.SCROLL_DOWN in events

    def test_scroll_up_pinky_thumb(self, fsm: PinchStateMachine) -> None:
        """Pinching pinky+thumb should produce SCROLL_UP events."""
        t = 0.0

        # First pinch → immediate scroll event
        events = fsm.update(_make_landmarks(thumb_pinky_dist=CLOSE), t)
        assert GestureEvent.SCROLL_UP in events

        # Holding for 60ms (> 50ms interval) → another scroll
        t += 0.060
        events = fsm.update(_make_landmarks(thumb_pinky_dist=CLOSE), t)
        assert GestureEvent.SCROLL_UP in events

    def test_scroll_stops_on_release(self, fsm: PinchStateMachine) -> None:
        """Releasing the ring/pinky pinch should stop scroll events."""
        t = 0.0

        # Start scrolling
        events = fsm.update(_make_landmarks(thumb_ring_dist=CLOSE), t)
        assert GestureEvent.SCROLL_DOWN in events

        # Release
        t += 0.060
        events = fsm.update(_make_landmarks(thumb_ring_dist=FAR), t)
        assert GestureEvent.SCROLL_DOWN not in events

        # Stay released
        t += 0.060
        events = fsm.update(_make_landmarks(thumb_ring_dist=FAR), t)
        assert GestureEvent.SCROLL_DOWN not in events


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
