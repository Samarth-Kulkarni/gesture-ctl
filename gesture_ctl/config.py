"""Central configuration for gesture-ctl.

All tunable parameters live here as a frozen dataclass so every module
shares a single source of truth.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field


# Ensure Windows gives us physical pixels, not scaled virtual pixels (e.g. 125% DPI)
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass


def _screen_width() -> int:
    """Return primary monitor width in physical pixels via Win32."""
    try:
        return ctypes.windll.user32.GetSystemMetrics(0)
    except Exception:
        return 1920


def _screen_height() -> int:
    """Return primary monitor height in physical pixels via Win32."""
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

    # ── EMA Smoothing (Adaptive) ─────────────────────────────────────────
    ema_alpha: float = 0.2
    ema_alpha_min: float = 0.02     # locks cursor completely still when hovering
    ema_alpha_max: float = 0.40     # smooths out tracking shakes when moving

    # ── Active Margin Box (pixels inside the camera frame) ──────────────
    margin_px: int = 160            # balances easy reach with lower noise magnification

    # ── Pinch Detection ─────────────────────────────────────────────────
    pinch_threshold: float = 0.040  # 3D normalised tip distance
    pinch_release_threshold: float = 0.060  # upper distance to release a pinch (hysteresis)
    drag_release_frames: int = 2    # consecutive unpinched frames needed to end drag
    quick_pinch_ms: int = 250       # max pinch duration for a single click
    double_pinch_window_ms: int = 350  # window for second pinch → double-click
    drag_hold_ms: int = 250         # hold duration to start drag

    # ── Scroll Gestures (Ring / Pinky + Thumb) ──────────────────────────
    scroll_threshold: float = 0.040     # pinch trigger distance for ring/pinky
    scroll_release_threshold: float = 0.060  # release distance for scroll pinch
    scroll_interval_ms: float = 100.0   # ms between scroll pulses while held
    scroll_step: int = 80               # scroll amount per pulse (smaller = smoother)

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
