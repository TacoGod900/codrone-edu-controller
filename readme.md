# Fingertip CoDrone EDU Controller

This project uses OpenCV and MediaPipe hand tracking to control a CoDrone EDU from a webcam feed. The main app is `drone_showcase.py`.

The controller can also run without a connected drone. If the CoDrone EDU controller is missing or pairing fails, the app switches to simulation mode so you can still test the camera, fingertip zones, hover feedback, and keyboard controls.

## Requirements

- Python 3.10 or newer recommended
- USB or laptop camera
- CoDrone EDU controller and drone, optional for real flight
- `src/hand_landmarker.task`, included in this project

Install dependencies from `requirements.txt`:

```shell
python -m venv .venv
.venv/Scripts/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with:

```shell
source .venv/bin/activate
```

## Project Structure

```text
drone/
|-- .gitignore
|-- drone_showcase.py
|-- readme.md
|-- requirements.txt
|-- dev/
|   |-- generate_marker.py
|   |-- get_calibration.py
|   |-- show_calibration.py
|   `-- img/
|       `-- pattern.png
`-- src/
    |-- hand_landmarker.task
    |-- photos/
    |   |-- monk_0.jpg
    |   |-- monk_1.jpg
    |   |-- treasure_1.jpg
    |   `-- treasure_2.jpg
```

Generated folders such as `__pycache__/`, `.venv/`, calibration output, and generated markers are not part of the intended source structure.

## Run

From the `drone/` folder:

```shell
python drone_showcase.py
```

If the drone is connected, the app will try to pair with it. If pairing fails, it prints a `[SIM]` message and continues without sending real drone commands.

## Controls

The app uses the index fingertip for screen zones. It also watches the thumb and index fingertip together for pinch rotation.

| Hand | Controls |
|------|----------|
| Left index fingertip | `Up`, `Down` |
| Right index fingertip | `Forward`, `Back`, `Left`, `Right` |

You can hover over two movement controls at the same time, for example `Up + Forward`. The window shows the current hover and command, and the terminal prints messages such as:

```text
[HOVER] Forward
[COMMAND] Up + Forward
[SIM] motion Up + Forward
```

To rotate, pinch your thumb and index fingertip together. The app marks where the pinch started, then draws an arrow from that start point to the current pinch position. Keep pinching and drag left or right:

| Gesture | Action |
|---------|--------|
| Pinch + drag left | Rotate left |
| Pinch + drag right | Rotate right |

While the pinch is active, normal movement zones are paused so the rotate gesture does not accidentally trigger forward/left/right movement.

## Top Buttons

Hold an index fingertip over a top button for about `0.6` seconds.

| Button | Action |
|--------|--------|
| `TAKEOFF` | Take off |
| `LAND` | Land |
| `STOP` | Stop and exit the program |

## Keyboard Fallbacks

| Key | Action |
|-----|--------|
| `T` | Takeoff |
| `Space` | Stop/pause |
| `L` | Land |
| `Q` or `Esc` | Quit |

## Simulation Mode

Simulation mode is used when:

- `codrone_edu` is not installed
- the CoDrone EDU controller is not plugged in
- pairing exits or fails

In simulation mode, the camera UI still works and the terminal prints `[SIM]` output instead of flying a drone.

## Camera Calibration Tools

The `dev/` folder contains helper scripts from the original OpenCV AR setup:

| Script | Purpose |
|--------|---------|
| `dev/get_calibration.py` | Create `src/camera_params.npz` from a printed checkerboard pattern. |
| `dev/show_calibration.py` | Display saved camera calibration values. |
| `dev/generate_marker.py` | Generate ArUco markers for experiments. |

The drone controller does not require camera calibration, but calibration can help if you add marker-based AR features later.
