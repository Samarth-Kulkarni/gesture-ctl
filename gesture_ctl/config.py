"""Central configuration for gesture-ctl.

All tunable parameters live here as a frozen dataclass so every module
shares a single source of truth.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field


def _screen_width() -> int:
    """Return primary monitor width in pixels via Win32."""
    try:
        return ctypes.windll.user32.GetSystemMetrics(0)
    except Exception:
        return 1920


def _screen_height() -> int:
    """Return primary monitor height in pixels via Win32."""
    try:
        return ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        return 1080


@dataclass
class Config:
    """Runtime-immutable configuration container."""

    # ── Camera ──────────────────────────────────────────────────────────
    cam_index: int = 0
    cam_width: int = 640
    cam_height: int = 480

    # ── EMA Smoothing ───────────────────────────────────────────────────
    ema_alpha: float = 0.2

    # ── Active Margin Box (pixels inside the camera frame) ──────────────
    margin_px: int = 100

    # ── Pinch Detection ─────────────────────────────────────────────────
    pinch_threshold: float = 0.045  # normalised 3-D distance
    quick_pinch_ms: int = 300       # max duration for a "click"
    double_pinch_window_ms: int = 400  # window for second pinch → dbl-click
    drag_hold_ms: int = 300         # min hold to start drag

    # ── System Gesture Thresholds ───────────────────────────────────────
    toggle_hold_s: float = 1.0      # V-sign hold to toggle engagement
    palm_hold_frames: int = 10      # open-palm frames for play/pause
    stationary_threshold: float = 0.015  # max landmark movement to count as "still"

    # ── Volume ──────────────────────────────────────────────────────────
    volume_min_dist: float = 0.03   # thumb-index dist → 0 %
    volume_max_dist: float = 0.25   # thumb-index dist → 100 %

    # ── Display (auto-detected) ─────────────────────────────────────────
    screen_width: int = field(default_factory=_screen_width)
    screen_height: int = field(default_factory=_screen_height)

    # ── Debug ───────────────────────────────────────────────────────────
    show_preview: bool = True
    fps_overlay: bool = True
