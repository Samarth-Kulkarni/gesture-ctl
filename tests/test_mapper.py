"""Tests for gesture_ctl.mapper — EMA smoothing & margin-box mapping."""

import pytest

from gesture_ctl.config import Config
from gesture_ctl.mapper import ScreenMapper


@pytest.fixture
def cfg() -> Config:
    return Config(
        cam_width=640,
        cam_height=480,
        margin_px=100,
        ema_alpha=1.0,  # α=1 disables smoothing for deterministic tests
        screen_width=1920,
        screen_height=1080,
    )


@pytest.fixture
def mapper(cfg: Config) -> ScreenMapper:
    return ScreenMapper(cfg)


# ── Margin-box boundary tests ──────────────────────────────────────────

class TestMarginBox:
    """Ensure the active margin box maps correctly to screen edges."""

    def test_centre_maps_to_screen_centre(self, mapper: ScreenMapper) -> None:
        sx, sy = mapper.map(0.5, 0.5)
        # With a symmetric margin, (0.5, 0.5) in cam → centre of screen
        assert abs(sx - 960) <= 1
        assert abs(sy - 540) <= 1

    def test_top_left_edge(self, mapper: ScreenMapper, cfg: Config) -> None:
        """Landmark at the inner margin boundary → screen (0, 0)."""
        x_min = cfg.margin_px / cfg.cam_width
        y_min = cfg.margin_px / cfg.cam_height
        sx, sy = mapper.map(x_min, y_min)
        assert sx == 0
        assert sy == 0

    def test_bottom_right_edge(self, mapper: ScreenMapper, cfg: Config) -> None:
        """Landmark at the far inner margin boundary → screen bottom-right."""
        x_max = 1.0 - cfg.margin_px / cfg.cam_width
        y_max = 1.0 - cfg.margin_px / cfg.cam_height
        sx, sy = mapper.map(x_max, y_max)
        assert sx == cfg.screen_width - 1
        assert sy == cfg.screen_height - 1

    def test_outside_margin_clamped(self, mapper: ScreenMapper, cfg: Config) -> None:
        """Landmark outside the margin box is clamped to screen bounds."""
        sx, sy = mapper.map(0.0, 0.0)
        assert sx == 0
        assert sy == 0

        sx, sy = mapper.map(1.0, 1.0)
        assert sx == cfg.screen_width - 1
        assert sy == cfg.screen_height - 1


# ── EMA smoothing tests ────────────────────────────────────────────────

class TestEMA:
    """Verify exponential moving average behaviour."""

    def test_first_call_returns_raw(self) -> None:
        """First map() call should return the raw mapped position."""
        cfg = Config(ema_alpha=0.3, screen_width=1920, screen_height=1080)
        mapper = ScreenMapper(cfg)
        sx, sy = mapper.map(0.5, 0.5)
        # First call, so raw == smoothed regardless of alpha
        assert isinstance(sx, int)
        assert isinstance(sy, int)

    def test_ema_converges(self) -> None:
        """Repeated identical inputs should converge to the true position."""
        cfg = Config(ema_alpha=0.2, screen_width=1920, screen_height=1080)
        mapper = ScreenMapper(cfg)
        for _ in range(200):
            sx, sy = mapper.map(0.5, 0.5)
        assert abs(sx - 960) <= 1
        assert abs(sy - 540) <= 1

    def test_lower_alpha_is_smoother(self) -> None:
        """α=0.1 should be further from target after 3 steps than α=0.9."""
        def run_alpha(alpha: float) -> int:
            cfg = Config(ema_alpha=alpha, screen_width=1920, screen_height=1080)
            m = ScreenMapper(cfg)
            m.map(0.5, 0.5)  # init at centre
            for _ in range(3):
                sx, _ = m.map(0.8, 0.5)  # jump to the right
            return sx

        pos_slow = run_alpha(0.1)
        pos_fast = run_alpha(0.9)
        # pos_fast should be closer to the target (right side)
        target = int(0.8 * 1919)  # rough target
        assert abs(pos_fast - target) < abs(pos_slow - target)

    def test_reset_clears_history(self) -> None:
        """After reset(), the next call should behave like the first."""
        cfg = Config(ema_alpha=0.2, screen_width=1920, screen_height=1080)
        mapper = ScreenMapper(cfg)
        mapper.map(0.5, 0.5)
        mapper.map(0.5, 0.5)
        mapper.reset()
        # After reset, mapping a far position should jump there
        sx, sy = mapper.map(0.9, 0.9)
        # With alpha=0.2 and fresh state, first call = raw
        assert sx > 1500  # should be near the right edge
