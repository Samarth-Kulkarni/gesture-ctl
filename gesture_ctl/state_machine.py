"""Phase 3 — Pinch-based finite state machine for click / drag / double-click.

Two independent FSMs run in parallel:

* **Left FSM** — Thumb Tip (4) ↔ Index Tip (8)
* **Right FSM** — Thumb Tip (4) ↔ Middle Tip (12)

Each follows the same transition logic::

    IDLE ──pinch──► PINCHED ──release < quick_ms──► CLICK_CANDIDATE
                       │                                │
                       │ held ≥ drag_ms                  │ re-pinch < dbl_ms
                       ▼                                 ▼
                   DRAGGING ──release──► IDLE        DOUBLE_CLICK ──► IDLE

The module exposes a single :class:`PinchStateMachine` that returns
:class:`GestureEvent` values every frame.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass

from gesture_ctl.config import Config
from gesture_ctl.tracker import LM


# ── Gesture events ──────────────────────────────────────────────────────

class GestureEvent(enum.Enum):
    """Discrete events produced by the pinch FSMs."""

    NONE = "none"

    LEFT_CLICK = "left_click"
    LEFT_DOUBLE_CLICK = "left_double_click"
    LEFT_DRAG_START = "left_drag_start"
    LEFT_DRAG_END = "left_drag_end"

    RIGHT_CLICK = "right_click"
    RIGHT_DOUBLE_CLICK = "right_double_click"
    RIGHT_DRAG_START = "right_drag_start"
    RIGHT_DRAG_END = "right_drag_end"


# ── Internal FSM states ────────────────────────────────────────────────

class _State(enum.Enum):
    IDLE = 0
    PINCHED = 1
    CLICK_CANDIDATE = 2
    DRAGGING = 3


# ── Single-finger FSM ──────────────────────────────────────────────────

class _PinchFSM:
    """State machine for one pinch pair (e.g. thumb+index)."""

    def __init__(
        self,
        cfg: Config,
        click_event: GestureEvent,
        double_click_event: GestureEvent,
        drag_start_event: GestureEvent,
        drag_end_event: GestureEvent,
    ) -> None:
        self._threshold = cfg.pinch_threshold
        self._quick_ms = cfg.quick_pinch_ms
        self._dbl_ms = cfg.double_pinch_window_ms
        self._drag_ms = cfg.drag_hold_ms

        self._click_ev = click_event
        self._dbl_ev = double_click_event
        self._drag_start_ev = drag_start_event
        self._drag_end_ev = drag_end_event

        self._state = _State.IDLE
        self._pinch_start: float = 0.0
        self._last_click: float | None = None

    def update(self, distance: float, now: float) -> GestureEvent:
        """Advance the FSM given the current pinch *distance* and timestamp.

        Parameters
        ----------
        distance : float
            3-D Euclidean distance between the two finger tips (normalised).
        now : float
            Current time in **seconds** (``time.perf_counter()``).
        """
        pinching = distance < self._threshold
        ms = lambda t: (now - t) * 1000.0  # noqa: E731

        if self._state is _State.IDLE:
            if pinching:
                self._pinch_start = now
                self._state = _State.PINCHED
            return GestureEvent.NONE

        if self._state is _State.PINCHED:
            if not pinching:
                # Released — was it quick enough for a click?
                if ms(self._pinch_start) < self._quick_ms:
                    # Check for double-click
                    if self._last_click is not None and ms(self._last_click) < self._dbl_ms:
                        self._state = _State.IDLE
                        self._last_click = None
                        return self._dbl_ev
                    # Instant single click on release (0ms artificial delay)
                    self._last_click = now
                    self._state = _State.IDLE
                    return self._click_ev
                else:
                    self._state = _State.IDLE
                    return GestureEvent.NONE
            elif ms(self._pinch_start) >= self._drag_ms:
                self._state = _State.DRAGGING
                return self._drag_start_ev
            return GestureEvent.NONE

        if self._state is _State.DRAGGING:
            if not pinching:
                self._state = _State.IDLE
                return self._drag_end_ev
            return GestureEvent.NONE

        return GestureEvent.NONE  # pragma: no cover


# ── 3D Euclidean distance helper ────────────────────────────────────────

def _dist3d(
    landmarks: list[tuple[float, float, float]],
    idx_a: int,
    idx_b: int,
) -> float:
    """3-D Euclidean distance between two landmarks."""
    ax, ay, az = landmarks[idx_a]
    bx, by, bz = landmarks[idx_b]
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


# ── Combined FSM facade ────────────────────────────────────────────────

class PinchStateMachine:
    """Runs left-click and right-click FSMs in parallel.

    Call :meth:`update` once per frame with the full landmark list.
    It returns a list of :class:`GestureEvent` values (typically 0–2).
    """

    def __init__(self, cfg: Config) -> None:
        self._left = _PinchFSM(
            cfg,
            click_event=GestureEvent.LEFT_CLICK,
            double_click_event=GestureEvent.LEFT_DOUBLE_CLICK,
            drag_start_event=GestureEvent.LEFT_DRAG_START,
            drag_end_event=GestureEvent.LEFT_DRAG_END,
        )
        self._right = _PinchFSM(
            cfg,
            click_event=GestureEvent.RIGHT_CLICK,
            double_click_event=GestureEvent.RIGHT_DOUBLE_CLICK,
            drag_start_event=GestureEvent.RIGHT_DRAG_START,
            drag_end_event=GestureEvent.RIGHT_DRAG_END,
        )

    def update(
        self,
        landmarks: list[tuple[float, float, float]],
        now: float | None = None,
    ) -> list[GestureEvent]:
        """Process one frame and return any generated events.

        Parameters
        ----------
        landmarks : list
            21-element list of ``(x, y, z)`` normalised landmarks.
        now : float, optional
            Timestamp; defaults to ``time.perf_counter()``.
        """
        if now is None:
            now = time.perf_counter()

        left_dist = _dist3d(landmarks, LM.THUMB_TIP, LM.INDEX_TIP)
        right_dist = _dist3d(landmarks, LM.THUMB_TIP, LM.MIDDLE_TIP)

        events: list[GestureEvent] = []

        ev_l = self._left.update(left_dist, now)
        if ev_l is not GestureEvent.NONE:
            events.append(ev_l)

        ev_r = self._right.update(right_dist, now)
        if ev_r is not GestureEvent.NONE:
            events.append(ev_r)

        return events
