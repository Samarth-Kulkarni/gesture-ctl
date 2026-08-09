# gesture-ctl

**Real-Time OS Gesture Controller & Touchless Air-Mouse for Windows**

Control your Windows PC with hand gestures — move the cursor, click, drag, adjust volume, and play/pause media — all touchless via your webcam.

---

## Features

| Gesture | Action |
|---------|--------|
| ☝️ **Raise Index Finger** | Move cursor (tracks Index Knuckle / Landmark 5) |
| 🤏 **Thumb + Index Pinch** (quick) | Left Click |
| 🤏🤏 **Double Pinch** (< 400ms) | Left Double-Click |
| 🤏 **Thumb + Index Hold** (> 300ms) | Left Click & Drag |
| 🤏 **Thumb + Middle Pinch** (quick) | Right Click |
| 🤏 **Thumb + Middle Hold** (> 300ms) | Right Click & Hold |
| ✌️ **V-Sign** (held 1s) | Master Engage / Disengage Toggle |
| 🖐️ **Open Palm** (held 10+ frames) | Play / Pause Media |

---

## Requirements

- **Windows 10/11**
- **Python 3.9–3.12** (MediaPipe does not support 3.13+)
- **Webcam** accessible via DirectShow

> ⚠️ **Kaspersky / Antivirus Users:** Webcam protection may block `gesture-ctl` from accessing the camera. Add the Python executable (or `gesture-ctl`) to your antivirus whitelist:
> - Kaspersky: *Settings → Privacy → Webcam → Add exclusion*

---

## Installation

```bash
# Clone or navigate to the project
cd gesture-ctl

# Create a virtual environment with Python 3.12
# (if you have Python 3.13+ as system default, use py launcher)
py -3.12 -m venv .venv
.venv\Scripts\activate

# Install in editable mode
pip install -e .

# (Optional) Install dev dependencies for testing
pip install -e ".[dev]"
```

If you don't have Python 3.12 alongside 3.13, install it via the
[Python Windows installer](https://www.python.org/downloads/release/python-3129/)
or use [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
.venv\Scripts\activate
uv pip install -e .
```

---

## Usage

### Start (foreground with preview window)
```bash
gesture-ctl start
```

### Start (no preview window)
```bash
gesture-ctl start --no-preview
```

### Select a different camera
```bash
gesture-ctl start --camera 1
```

### Tune EMA smoothing (lower = smoother, higher = snappier)
```bash
gesture-ctl start --ema-alpha 0.15
```

### Check status
```bash
gesture-ctl status
```

### Stop
```bash
gesture-ctl stop
```

### Debug logging
```bash
gesture-ctl start -v
```

---

## How It Works

```
Webcam (OpenCV) → Hand Tracker (MediaPipe) → Screen Mapper (EMA)
                                             ↓
                                       Pinch State Machine
                                             ↓
                                      OS Dispatcher (Win32 / pyautogui)
```

1. **Capture** — OpenCV reads frames at 640×480 via DirectShow with 1-frame buffer for low latency.
2. **Track** — MediaPipe Hands extracts 21 3D landmarks using the lite model (CPU optimised).
3. **Map** — An Active Margin Box maps the inner camera region to full screen resolution. An EMA filter (α ≈ 0.2) smooths cursor jitter.
4. **Classify** — A finite state machine distinguishes single-click, double-click, and drag-and-hold based on pinch duration and timing.
5. **Dispatch** — `win32api.SetCursorPos` moves the cursor; `pyautogui` handles clicks; `pynput` triggers media keys.

---

## Project Structure

```
gesture-ctl/
├── pyproject.toml
├── README.md
├── gesture_ctl/
│   ├── __init__.py       # Package version
│   ├── config.py         # Central configuration dataclass
│   ├── capture.py        # OpenCV camera wrapper
│   ├── tracker.py        # MediaPipe hand landmark tracker
│   ├── mapper.py         # Screen coordinate mapping + EMA
│   ├── gestures.py       # Finger-state classifiers (V-sign, palm, fist)
│   ├── state_machine.py  # Pinch-based click/drag FSM
│   ├── dispatcher.py     # OS action executor (cursor, clicks, media)
│   ├── engine.py         # Main orchestration loop
│   ├── watchdog.py       # CPU load watchdog
│   └── cli.py            # CLI entry point (click)
└── tests/
    ├── test_mapper.py
    ├── test_state_machine.py
    └── test_gestures.py
```

---

## License

MIT
