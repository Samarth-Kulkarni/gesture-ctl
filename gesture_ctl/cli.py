"""Phase 5 — CLI entry-point for gesture-ctl.

Subcommands
-----------
    gesture-ctl start [--no-preview] [--camera N] [--ema-alpha F]
    gesture-ctl stop
    gesture-ctl status
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

import click
import psutil

from gesture_ctl import __version__
from gesture_ctl.config import Config

_PID_FILE = Path.home() / ".gesture-ctl.pid"

logger = logging.getLogger("gesture_ctl")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(__version__, prog_name="gesture-ctl")
def main() -> None:
    """Real-Time OS Gesture Controller & Touchless Air-Mouse."""


@main.command()
@click.option("--no-preview", is_flag=True, help="Disable the debug camera window.")
@click.option("--camera", default=0, type=int, help="Camera device index.")
@click.option("--ema-alpha", default=0.2, type=float, help="EMA smoothing factor (0–1).")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def start(no_preview: bool, camera: int, ema_alpha: float, verbose: bool) -> None:
    """Launch the gesture-control engine."""
    _setup_logging(verbose)

    cfg = Config(
        cam_index=camera,
        ema_alpha=ema_alpha,
        show_preview=not no_preview,
    )

    # Write PID for `stop` command
    _PID_FILE.write_text(str(os.getpid()))

    # Lazy import to keep CLI snappy
    from gesture_ctl.engine import GestureEngine

    engine = GestureEngine(cfg)

    def _handle_signal(signum, frame):  # noqa: ARG001
        logger.info("Received signal %s — shutting down …", signum)
        engine.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        engine.run()
    finally:
        if _PID_FILE.exists():
            _PID_FILE.unlink(missing_ok=True)


@main.command()
def stop() -> None:
    """Stop a running gesture-ctl instance."""
    if not _PID_FILE.exists():
        click.echo("No running instance found (PID file missing).")
        sys.exit(1)

    pid = int(_PID_FILE.read_text().strip())

    if not psutil.pid_exists(pid):
        click.echo(f"PID {pid} is not running. Cleaning up stale PID file.")
        _PID_FILE.unlink(missing_ok=True)
        sys.exit(1)

    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=5)
        click.echo(f"gesture-ctl (PID {pid}) stopped.")
    except psutil.TimeoutExpired:
        proc.kill()
        click.echo(f"gesture-ctl (PID {pid}) killed (did not stop gracefully).")
    finally:
        _PID_FILE.unlink(missing_ok=True)


@main.command()
def status() -> None:
    """Check whether gesture-ctl is running."""
    if not _PID_FILE.exists():
        click.echo("gesture-ctl is NOT running.")
        sys.exit(1)

    pid = int(_PID_FILE.read_text().strip())

    if psutil.pid_exists(pid):
        proc = psutil.Process(pid)
        click.echo(
            f"gesture-ctl is RUNNING  (PID {pid}, "
            f"CPU {proc.cpu_percent(interval=0.5):.1f}%, "
            f"Mem {proc.memory_info().rss / 1024 / 1024:.1f} MB)"
        )
    else:
        click.echo(f"PID {pid} is stale — cleaning up.")
        _PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
