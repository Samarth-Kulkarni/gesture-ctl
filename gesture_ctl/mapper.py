"""Phase 2 — Screen coordinate mapping with EMA anti-jitter filtering.

Maps a normalised landmark position (0..1 within the camera frame) to an
absolute pixel coordinate on the display, using an *Active Margin Box* so
the user doesn't need to reach the very edge of the camera FOV to hit
screen corners.

An Exponential Moving Average (EMA) filter is applied independently to
x and y to eliminate micro-jitter while keeping latency under ~15 ms.
"""

from __future__ import annotations

from gesture_ctl.config import Config


class ScreenMapper:
    """Map normalised camera-space coordinates → smoothed screen pixels.

    Active Margin Box
    -----------------
    The camera frame (e.g. 640×480) is logically shrunk by ``margin_px``
    on each side.  Any landmark inside this inner rectangle maps linearly
    to 0 %–100 % of the display resolution; positions outside are clamped.

    EMA Smoothing
    -------------
    ``smoothed = α · raw + (1 − α) · prev``

    Lower α → smoother but laggier; higher α → responsive but jittery.
    ``α ≈ 0.2`` is a good default for 60 FPS capture.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

        # Pre-compute active-region boundaries (normalised 0..1)
        self._x_min = cfg.margin_px / cfg.cam_width
        self._x_max = 1.0 - self._x_min
        self._y_min = cfg.margin_px / cfg.cam_height
        self._y_max = 1.0 - self._y_min

        # EMA state
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None

    # ── Public API ──────────────────────────────────────────────────────

    def map(self, norm_x: float, norm_y: float) -> tuple[int, int]:
        """Convert a normalised landmark to smoothed screen pixel coords.

        Parameters
        ----------
        norm_x, norm_y : float
            Normalised coordinates in ``[0, 1]`` from MediaPipe.

        Returns
        -------
        (screen_x, screen_y) : tuple[int, int]
            Absolute pixel position clamped to the display bounds.
        """
        # 1. Map to 0..1 within the active margin box
        rel_x = (norm_x - self._x_min) / (self._x_max - self._x_min)
        rel_y = (norm_y - self._y_min) / (self._y_max - self._y_min)

        # Clamp to [0, 1]
        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))

        # 2. Scale to screen resolution
        raw_x = rel_x * (self._cfg.screen_width - 1)
        raw_y = rel_y * (self._cfg.screen_height - 1)

        # 3. Adaptive EMA smoothing based on movement velocity
        if self._smooth_x is None or self._smooth_y is None:
            self._smooth_x = raw_x
            self._smooth_y = raw_y
        else:
            dx = raw_x - self._smooth_x
            dy = raw_y - self._smooth_y
            dist = (dx * dx + dy * dy) ** 0.5

            # Dynamically scale alpha: 0.02 for small jitter, up to 0.40 for fast moves
            # dist threshold: < 30px -> min alpha (cursor locked), 100px+ -> max alpha
            speed_ratio = min(1.0, max(0.0, (dist - 30.0) / 70.0))
            alpha = self._cfg.ema_alpha_min + speed_ratio * (self._cfg.ema_alpha_max - self._cfg.ema_alpha_min)

            self._smooth_x = alpha * raw_x + (1.0 - alpha) * self._smooth_x
            self._smooth_y = alpha * raw_y + (1.0 - alpha) * self._smooth_y

        return int(round(self._smooth_x)), int(round(self._smooth_y))

    def reset(self) -> None:
        """Clear EMA history (e.g. when hand re-enters the frame)."""
        self._smooth_x = None
        self._smooth_y = None
