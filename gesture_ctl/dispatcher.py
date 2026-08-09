"""Phase 4 — OS action dispatcher.

Translates :class:`GestureEvent` values and screen coordinates into
real Windows OS actions: cursor moves, clicks, drags, and media keys.

Cursor positioning uses ``win32api.SetCursorPos`` via ctypes for minimal
latency (~0.1 ms).  Click/drag events use ``pyautogui``.  Media keys
use ``pynput``.
"""

from __future__ import annotations

import ctypes
import logging

import pyautogui
from pynput.keyboard import Controller as KbController, Key

from gesture_ctl.state_machine import GestureEvent

logger = logging.getLogger(__name__)

# Disable pyautogui pauses for real-time use
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False  # disabled so cursor can safely hit the edges of the screen


# ── Low-level cursor positioning ────────────────────────────────────────

def _set_cursor(x: int, y: int) -> None:
    """Move the OS cursor via Win32 ``SetCursorPos``."""
    try:
        ctypes.windll.user32.SetCursorPos(x, y)
    except Exception:
        pyautogui.moveTo(x, y, _pause=False)


# ── Dispatcher class ────────────────────────────────────────────────────

class Dispatcher:
    """Execute OS-level actions from gesture events."""

    def __init__(self) -> None:
        self._keyboard = KbController()

    # ── Cursor ──────────────────────────────────────────────────────────

    def move_cursor(self, x: int, y: int) -> None:
        """Instantly move the OS cursor to *(x, y)*."""
        _set_cursor(x, y)

    # ── Click / drag ────────────────────────────────────────────────────

    def execute_event(self, event: GestureEvent) -> None:
        """Dispatch a single :class:`GestureEvent` to the OS."""
        match event:
            case GestureEvent.NONE:
                return

            # Left
            case GestureEvent.LEFT_CLICK:
                logger.debug("LEFT CLICK")
                pyautogui.click(button="left")
            case GestureEvent.LEFT_DOUBLE_CLICK:
                logger.debug("LEFT DOUBLE-CLICK")
                pyautogui.doubleClick(button="left")
            case GestureEvent.LEFT_DRAG_START:
                logger.debug("LEFT DRAG START")
                pyautogui.mouseDown(button="left")
            case GestureEvent.LEFT_DRAG_END:
                logger.debug("LEFT DRAG END")
                pyautogui.mouseUp(button="left")

            # Right
            case GestureEvent.RIGHT_CLICK:
                logger.debug("RIGHT CLICK")
                pyautogui.click(button="right")
            case GestureEvent.RIGHT_DOUBLE_CLICK:
                logger.debug("RIGHT DOUBLE-CLICK")
                pyautogui.doubleClick(button="right")
            case GestureEvent.RIGHT_DRAG_START:
                logger.debug("RIGHT DRAG START")
                pyautogui.mouseDown(button="right")
            case GestureEvent.RIGHT_DRAG_END:
                logger.debug("RIGHT DRAG END")
                pyautogui.mouseUp(button="right")

    # ── Media keys ──────────────────────────────────────────────────────

    def play_pause(self) -> None:
        """Simulate media play/pause key press."""
        logger.debug("MEDIA PLAY/PAUSE")
        self._keyboard.press(Key.media_play_pause)
        self._keyboard.release(Key.media_play_pause)

    def volume_up(self) -> None:
        """One step of volume up."""
        self._keyboard.press(Key.media_volume_up)
        self._keyboard.release(Key.media_volume_up)

    def volume_down(self) -> None:
        """One step of volume down."""
        self._keyboard.press(Key.media_volume_down)
        self._keyboard.release(Key.media_volume_down)
