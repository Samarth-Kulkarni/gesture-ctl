"""Phase 5 — Main orchestration engine.

Runs the real-time pipeline:

  Camera → Tracker → Mapper → StateMachine → Dispatcher

Also handles the master engagement toggle (V-sign), play/pause
(open palm), and provides a debug preview window.
"""

from __future__ import annotations

import logging
import time

import cv2

from gesture_ctl.capture import CameraCapture
from gesture_ctl.config import Config
from gesture_ctl.dispatcher import Dispatcher
from gesture_ctl.gestures import (
    StationaryDetector,
    is_open_palm,
    is_v_sign,
)
from gesture_ctl.mapper import ScreenMapper
from gesture_ctl.state_machine import GestureEvent, PinchStateMachine
from gesture_ctl.tracker import HandTracker, LM
from gesture_ctl.watchdog import Watchdog

logger = logging.getLogger(__name__)

_PREVIEW_WINDOW = "gesture-ctl preview"


class GestureEngine:
    """Top-level controller that wires every subsystem together."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._camera = CameraCapture(cfg)
        self._tracker = HandTracker()
        self._mapper = ScreenMapper(cfg)
        self._fsm = PinchStateMachine(cfg)
        self._dispatcher = Dispatcher(cfg)
        self._watchdog = Watchdog(cam_index=cfg.cam_index)

        # System gesture state
        self._engaged = False
        self._v_detector = StationaryDetector(cfg, [LM.INDEX_TIP, LM.MIDDLE_TIP])
        self._palm_detector = StationaryDetector(cfg, [LM.WRIST, LM.INDEX_TIP])
        self._v_start: float | None = None
        self._toggle_cooldown = 0.0
        self._palm_count = 0
        self._play_pause_cooldown = 0.0

        self._running = False

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the real-time gesture-control loop.  Blocks until stopped."""
        self._running = True
        logger.info("Starting gesture-ctl engine …")

        with self._camera, self._tracker:
            logger.info(
                "Camera opened at %dx%d  |  Screen %dx%d",
                self._cfg.cam_width,
                self._cfg.cam_height,
                self._cfg.screen_width,
                self._cfg.screen_height,
            )
            self._loop()

        if self._cfg.show_preview:
            cv2.destroyAllWindows()
        logger.info("Engine stopped.")

    def stop(self) -> None:
        """Signal the main loop to exit."""
        self._running = False

    # ── Internal loop ───────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            # Watchdog: pause if CPU is melting or camera is grabbed
            if self._watchdog.should_pause():
                logger.warning("Watchdog triggered — pausing …")
                self._watchdog.wait_until_clear()
                logger.info("Watchdog cleared — resuming.")
                continue

            frame, ts = self._camera.read_frame()
            if frame is None:
                logger.warning("Blank frame — retrying …")
                continue

            # Mirror the frame so hand movements feel natural
            frame = cv2.flip(frame, 1)

            result = self._tracker.process(frame)

            if result.detected:
                lms = result.landmarks
                hand = result.handedness

                # ── System gestures (always checked) ────────────────────
                v_active = is_v_sign(lms, hand)
                self._check_toggle(lms, hand, ts)

                if self._engaged:
                    self._check_play_pause(lms, hand, ts)

                    # ── Cursor movement ─────────────────────────────────
                    # We track the Index Knuckle (MCP) instead of the Tip.
                    # The tip curls downwards when pinching causing cursor drop,
                    # whereas the knuckle remains completely stable.
                    ix, iy = lms[LM.INDEX_MCP][0], lms[LM.INDEX_MCP][1]
                    sx, sy = self._mapper.map(ix, iy)
                    self._dispatcher.move_cursor(sx, sy)

                    # ── Click / drag / scroll FSM ────────────────────────
                    # Skip FSM while V-sign is held — curled ring/pinky
                    # would otherwise trigger accidental scroll events.
                    if not v_active:
                        events = self._fsm.update(lms, ts)
                        for ev in events:
                            self._dispatcher.execute_event(ev)
            else:
                # Hand lost — reset EMA so cursor doesn't jump when re-detected
                self._mapper.reset()

            # ── Debug preview ───────────────────────────────────────────
            if self._cfg.show_preview:
                self._draw_overlay(frame, result, ts)
                cv2.imshow(_PREVIEW_WINDOW, frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    self._running = False

    # ── Toggle engagement (V-sign held 1 s) ─────────────────────────────

    def _check_toggle(
        self,
        lms: list[tuple[float, float, float]],
        handedness: str,
        ts: float,
    ) -> None:
        if ts - self._toggle_cooldown < 2.0:
            return  # debounce — ignore V-sign right after a toggle

        if is_v_sign(lms, handedness):
            self._v_detector.push(lms)
            if self._v_start is None:
                self._v_start = ts
            elif ts - self._v_start >= self._cfg.toggle_hold_s:
                if self._v_detector.is_stationary(
                    int(self._cfg.toggle_hold_s * 30)  # ~30 frames at 30 fps
                ):
                    self._engaged = not self._engaged
                    state_str = "ENGAGED" if self._engaged else "DISENGAGED"
                    logger.info("Master toggle → %s", state_str)
                    self._v_start = None
                    self._v_detector.clear()
                    self._toggle_cooldown = ts
        else:
            self._v_start = None

    # ── Play / pause (open palm held > N frames) ────────────────────────

    def _check_play_pause(
        self,
        lms: list[tuple[float, float, float]],
        handedness: str,
        ts: float,
    ) -> None:
        if ts - self._play_pause_cooldown < 1.5:
            return  # debounce

        if is_open_palm(lms, handedness):
            self._palm_detector.push(lms)
            self._palm_count += 1
            if self._palm_count >= self._cfg.palm_hold_frames:
                if self._palm_detector.is_stationary(self._cfg.palm_hold_frames):
                    self._dispatcher.play_pause()
                    self._palm_count = 0
                    self._palm_detector.clear()
                    self._play_pause_cooldown = ts
        else:
            self._palm_count = 0

    # ── Debug overlay ───────────────────────────────────────────────────

    def _draw_overlay(self, frame, result, ts) -> None:
        """Draw FPS and engagement status onto the debug preview."""
        fps = self._camera.fps
        colour = (0, 255, 0) if self._engaged else (0, 0, 255)
        status = "ENGAGED" if self._engaged else "DISENGAGED"

        cv2.putText(
            frame,
            f"FPS: {fps:.0f}  |  {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colour,
            2,
        )

        if result.detected:
            self._tracker.draw_landmarks(frame, result)
