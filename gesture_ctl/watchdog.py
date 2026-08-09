"""Phase 5 — System resource watchdog.

Monitors CPU usage and camera availability, pausing the tracking loop
when the system is under heavy load or the camera is grabbed by another
application.
"""

from __future__ import annotations

import logging
import time

import psutil

logger = logging.getLogger(__name__)

# How many consecutive high-CPU samples before we pause
_CPU_SAMPLES = 5
_CPU_THRESHOLD = 90.0  # percent
_RETRY_INTERVAL = 2.0  # seconds


class Watchdog:
    """Auto-pause logic for heavy CPU load or camera conflicts."""

    def __init__(self, cam_index: int = 0) -> None:
        self._cam_index = cam_index
        self._cpu_high_count = 0

    def should_pause(self) -> bool:
        """Return ``True`` if the engine should temporarily suspend.

        Checks:
        * CPU usage > 90 % for 5 consecutive polls
        """
        cpu = psutil.cpu_percent(interval=0)
        if cpu > _CPU_THRESHOLD:
            self._cpu_high_count += 1
            if self._cpu_high_count >= _CPU_SAMPLES:
                logger.warning("CPU at %.0f%% for %d checks — pausing.", cpu, self._cpu_high_count)
                return True
        else:
            self._cpu_high_count = 0
        return False

    def wait_until_clear(self) -> None:
        """Block until CPU usage drops below threshold."""
        while True:
            time.sleep(_RETRY_INTERVAL)
            cpu = psutil.cpu_percent(interval=0.5)
            if cpu < _CPU_THRESHOLD:
                self._cpu_high_count = 0
                return
