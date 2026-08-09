"""Phase 1 — High-FPS webcam capture with OpenCV.

Provides a context-managed camera wrapper that reads frames at 640×480
and tracks real-time FPS via a rolling window.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from gesture_ctl.config import Config


class CameraCapture:
    """Thin wrapper around ``cv2.VideoCapture`` tuned for low-latency capture.

    Usage::

        cfg = Config()
        with CameraCapture(cfg) as cam:
            while True:
                frame, ts = cam.read_frame()
                if frame is None:
                    break
                print(f"FPS: {cam.fps:.1f}")
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._ts_window: deque[float] = deque(maxlen=30)

    # ── Context manager ─────────────────────────────────────────────────

    def open(self) -> None:
        """Open the camera and apply desired settings."""
        self._cap = cv2.VideoCapture(self._cfg.cam_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self._cfg.cam_index}. "
                "Check that no other application is using the webcam "
                "(Kaspersky webcam protection may block access — "
                "whitelist gesture-ctl in Settings → Privacy → Webcam)."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.cam_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.cam_height)
        self._cap.set(cv2.CAP_PROP_FPS, 60)
        # Reduce internal buffering for lower latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def close(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── Frame reading ───────────────────────────────────────────────────

    def read_frame(self) -> tuple[Optional[np.ndarray], float]:
        """Read a single BGR frame.

        Returns:
            ``(frame, timestamp)`` where *frame* is ``None`` on failure.
        """
        if self._cap is None:
            raise RuntimeError("Camera not opened. Use open() or a context manager.")

        ret, frame = self._cap.read()
        now = time.perf_counter()

        if not ret or frame is None:
            return None, now

        self._ts_window.append(now)
        return frame, now

    # ── FPS helpers ─────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        """Rolling FPS based on the last 30 frame timestamps."""
        if len(self._ts_window) < 2:
            return 0.0
        elapsed = self._ts_window[-1] - self._ts_window[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._ts_window) - 1) / elapsed

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
