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

Scroll gestures use dedicated FSMs:

* **Scroll Down FSM** — Thumb Tip (4) ↔ Ring Tip (16)
* **Scroll Up FSM**   — Thumb Tip (4) ↔ Pinky Tip (20)

Pinch detection uses **dual-threshold hysteresis**: a pinch activates when
distance drops below ``pinch_threshold`` and only deactivates when distance
rises above ``pinch_release_threshold``.  Drag release additionally requires
``drag_release_frames`` consecutive frames above the release threshold to
prevent accidental drops from landmark jitter.

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

    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"


# ── Internal FSM states ────────────────────────────────────────────────

class _State(enum.Enum):
    IDLE = 0
    PINCHED = 1
    CLICK_CANDIDATE = 2
    DRAGGING = 3


# ── Single-finger FSM (with hysteresis & release debounce) ─────────────

class _PinchFSM:
    """State machine for one pinch pair (e.g. thumb+index).

    Uses dual-threshold hysteresis: enters pinch at ``pinch_threshold``,
    exits only at ``pinch_release_threshold``.  Drag release additionally
    requires ``drag_release_frames`` consecutive unpinched frames.
    """

    def __init__(
        self,
        cfg: Config,
        click_event: GestureEvent,
        double_click_event: GestureEvent,
        drag_start_event: GestureEvent,
        drag_end_event: GestureEvent,
    ) -> None:
        self._threshold = cfg.pinch_threshold
        self._release_threshold = cfg.pinch_release_threshold
        self._quick_ms = cfg.quick_pinch_ms
        self._dbl_ms = cfg.double_pinch_window_ms
        self._drag_ms = cfg.drag_hold_ms
        self._drag_release_frames = cfg.drag_release_frames

        self._click_ev = click_event
        self._dbl_ev = double_click_event
        self._drag_start_ev = drag_start_event
        self._drag_end_ev = drag_end_event

        self._state = _State.IDLE
        self._pinch_start: float = 0.0
        self._last_click: float | None = None
        self._unpinch_count: int = 0  # consecutive frames above release threshold

    def update(self, distance: float, now: float) -> GestureEvent:
        """Advance the FSM given the current pinch *distance* and timestamp.

        Parameters
        ----------
        distance : float
            3-D Euclidean distance between the two finger tips (normalised).
        now : float
            Current time in **seconds** (``time.perf_counter()``).
        """
        # Hysteresis: use lower threshold to enter, upper to exit
        entering_pinch = distance < self._threshold
        still_pinched = distance < self._release_threshold
        ms = lambda t: (now - t) * 1000.0  # noqa: E731

        if self._state is _State.IDLE:
            if entering_pinch:
                self._pinch_start = now
                self._state = _State.PINCHED
                self._unpinch_count = 0
            return GestureEvent.NONE

        if self._state is _State.PINCHED:
            if not still_pinched:
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
                self._unpinch_count = 0
                return self._drag_start_ev
            return GestureEvent.NONE

        if self._state is _State.DRAGGING:
            if not still_pinched:
                self._unpinch_count += 1
                if self._unpinch_count >= self._drag_release_frames:
                    self._state = _State.IDLE
                    self._unpinch_count = 0
                    return self._drag_end_ev
            else:
                self._unpinch_count = 0
            return GestureEvent.NONE

        return GestureEvent.NONE  # pragma: no cover


# ── Scroll FSM (continuous scroll while pinched) ───────────────────────

class _ScrollFSM:
    """Emits scroll events repeatedly while a finger pair is pinched.

    Uses the same hysteresis approach as ``_PinchFSM``.
    """

    def __init__(
        self,
        cfg: Config,
        scroll_event: GestureEvent,
    ) -> None:
        self._threshold = cfg.scroll_threshold
        self._release_threshold = cfg.scroll_release_threshold
        self._interval_s = cfg.scroll_interval_ms / 1000.0
        self._scroll_ev = scroll_event

        self._active = False
        self._last_scroll: float = 0.0

    def update(self, distance: float, now: float) -> GestureEvent:
        """Return a scroll event if enough time has elapsed since the last pulse."""
        entering = distance < self._threshold
        still_held = distance < self._release_threshold

        if not self._active:
            if entering:
                self._active = True
                self._last_scroll = now
                return self._scroll_ev
        else:
            if not still_held:
                self._active = False
                return GestureEvent.NONE
            # Emit repeated scroll pulses
            if now - self._last_scroll >= self._interval_s:
                self._last_scroll = now
                return self._scroll_ev

        return GestureEvent.NONE


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
    """Runs left-click, right-click, and scroll FSMs in parallel.

    Call :meth:`update` once per frame with the full landmark list.
    It returns a list of :class:`GestureEvent` values (typically 0–4).
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
        self._scroll_down = _ScrollFSM(cfg, scroll_event=GestureEvent.SCROLL_DOWN)
        self._scroll_up = _ScrollFSM(cfg, scroll_event=GestureEvent.SCROLL_UP)

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
        ring_dist = _dist3d(landmarks, LM.THUMB_TIP, LM.RING_TIP)
        pinky_dist = _dist3d(landmarks, LM.THUMB_TIP, LM.PINKY_TIP)

        events: list[GestureEvent] = []

        ev_l = self._left.update(left_dist, now)
        if ev_l is not GestureEvent.NONE:
            events.append(ev_l)

        ev_r = self._right.update(right_dist, now)
        if ev_r is not GestureEvent.NONE:
            events.append(ev_r)

        ev_sd = self._scroll_down.update(ring_dist, now)
        if ev_sd is not GestureEvent.NONE:
            events.append(ev_sd)

        ev_su = self._scroll_up.update(pinky_dist, now)
        if ev_su is not GestureEvent.NONE:
            events.append(ev_su)

        return events
