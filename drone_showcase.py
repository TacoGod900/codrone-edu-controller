from dataclasses import dataclass
from os.path import abspath, dirname, join
from queue import Empty, Queue
from sys import exit
from threading import Thread
import time
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

try:
    from codrone_edu.drone import Drone
except ImportError:
    Drone = None


WINDOW_WIDTH = 1152
WINDOW_HEIGHT = 720
FPS = 30
WIN_NAME = "Fingertip Zone Drone Control"
MODEL_PATH = join(dirname(abspath(__file__)), "src", "hand_landmarker.task")

THUMB_TIP_ID = 4
INDEX_TIP_ID = 8
TIP_SMOOTHING = 0.45
ZONE_DEBOUNCE_S = 0.13
ICON_DWELL_S = 0.60
COMMAND_TICK_S = 0.12
STEP_COOLDOWN_S = 0.20
STEP_DISTANCE_M = 0.08
STEP_DURATION_S = 0.18
CONTROL_POWER = 25
YAW_POWER = 30
PINCH_ENTER_PX = 48
PINCH_EXIT_PX = 72
PINCH_DRAG_THRESHOLD_PX = 38
MOTION_NAMES = ("Up", "Down", "Forward", "Back", "Left", "Right", "YawLeft", "YawRight")
MOTION_LABELS = {
    "YawLeft": "Rotate Left",
    "YawRight": "Rotate Right",
}

COL_BG = (22, 24, 28)
COL_GRID = (245, 245, 245)
COL_DEAD = (90, 90, 90)
COL_LEFT = (75, 210, 250)
COL_RIGHT = (105, 190, 255)
COL_ACTIVE = (70, 235, 110)
COL_WARN = (60, 90, 255)
COL_TEXT = (245, 245, 245)
COL_DIM = (175, 175, 175)
COL_LEFT_TIP = (0, 255, 255)
COL_RIGHT_TIP = (255, 185, 80)


@dataclass(frozen=True)
class Rect:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    hand: Optional[str]
    symbol: str
    color: Tuple[int, int, int]

    def contains(self, x: int, y: int, pad: int = 0) -> bool:
        return self.x1 - pad <= x <= self.x2 + pad and self.y1 - pad <= y <= self.y2 + pad


@dataclass
class Fingertip:
    hand: str
    idx: int
    x: int
    y: int


@dataclass
class PinchInfo:
    hand: str
    thumb: Tuple[int, int]
    index: Tuple[int, int]
    anchor: Tuple[int, int]
    mid: Tuple[int, int]
    distance: float
    active: bool
    drag_dx: int
    motion: Optional[str]


class ZoneDebouncer:
    def __init__(self, hold_s: float):
        self.hold_s = hold_s
        self.candidate = None
        self.candidate_since = 0.0
        self.active = None

    def update(self, candidate, now: float):
        if candidate != self.candidate:
            self.candidate = candidate
            self.candidate_since = now

        if candidate is None:
            self.active = None
        elif now - self.candidate_since >= self.hold_s:
            self.active = candidate

        return self.active


class IconDwell:
    def __init__(self, dwell_s: float):
        self.dwell_s = dwell_s
        self.hovering = None
        self.started = 0.0
        self.fired_for = None

    def update(self, icon_name: Optional[str], now: float):
        if icon_name != self.hovering:
            self.hovering = icon_name
            self.started = now
            self.fired_for = None
            return None, 0.0

        if icon_name is None:
            return None, 0.0

        progress = min(1.0, (now - self.started) / self.dwell_s)
        if progress >= 1.0 and self.fired_for != icon_name:
            self.fired_for = icon_name
            return icon_name, progress

        return None, progress


class FingertipSmoother:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.points = {}

    def update(self, raw_tips: List[Fingertip]) -> List[Fingertip]:
        fresh_keys = set()
        smoothed = []

        for tip in raw_tips:
            key = (tip.hand, tip.idx)
            fresh_keys.add(key)
            prev = self.points.get(key)
            if prev is None:
                sx, sy = float(tip.x), float(tip.y)
            else:
                sx = prev[0] * (1.0 - self.alpha) + tip.x * self.alpha
                sy = prev[1] * (1.0 - self.alpha) + tip.y * self.alpha
            self.points[key] = (sx, sy)
            smoothed.append(Fingertip(tip.hand, tip.idx, int(sx), int(sy)))

        for key in list(self.points):
            if key not in fresh_keys:
                del self.points[key]

        return smoothed


class PinchTracker:
    def __init__(self):
        self.active_hand = None
        self.anchor = (0, 0)

    def update(self, points: List[Fingertip]) -> Tuple[Optional[PinchInfo], Optional[Tuple[str, ...]]]:
        by_hand = {}
        for point in points:
            by_hand.setdefault(point.hand, {})[point.idx] = point

        candidates = []
        for hand, hand_points in by_hand.items():
            thumb = hand_points.get(THUMB_TIP_ID)
            index = hand_points.get(INDEX_TIP_ID)
            if thumb is None or index is None:
                continue

            distance = float(np.hypot(index.x - thumb.x, index.y - thumb.y))
            mid = ((thumb.x + index.x) // 2, (thumb.y + index.y) // 2)
            candidates.append((hand, thumb, index, mid, distance))

        active_candidate = None
        if self.active_hand:
            active_candidate = next((c for c in candidates if c[0] == self.active_hand), None)
            if active_candidate is None or active_candidate[4] > PINCH_EXIT_PX:
                self.active_hand = None
                active_candidate = None

        if self.active_hand is None:
            close_candidates = [c for c in candidates if c[4] <= PINCH_ENTER_PX]
            if close_candidates:
                active_candidate = min(close_candidates, key=lambda c: c[4])
                self.active_hand = active_candidate[0]
                self.anchor = active_candidate[3]

        if active_candidate is None:
            return None, None

        hand, thumb, index, mid, distance = active_candidate
        drag_dx = mid[0] - self.anchor[0]
        motion = None
        if drag_dx <= -PINCH_DRAG_THRESHOLD_PX:
            motion = "YawLeft"
        elif drag_dx >= PINCH_DRAG_THRESHOLD_PX:
            motion = "YawRight"

        info = PinchInfo(
            hand=hand,
            thumb=(thumb.x, thumb.y),
            index=(index.x, index.y),
            anchor=self.anchor,
            mid=mid,
            distance=distance,
            active=True,
            drag_dx=drag_dx,
            motion=motion,
        )
        return info, (motion,) if motion else None


class DroneController:
    def __init__(self):
        self.drone = None
        self.state = "sim" if Drone is None else "grounded"
        self.queue = Queue()
        self.worker = Thread(target=self._run, daemon=True)
        self.running = True
        self.current_motion = None
        self.motion_token = 0
        self.last_step_at = 0.0
        self.worker.start()

    def pair(self):
        if Drone is None:
            print("[SIM] codrone_edu is not installed. Running without a drone.")
            return
        try:
            self.drone = Drone()
            self.drone.pair()
            print("[DRONE] Paired. Use the takeoff icon or press T.")
        except SystemExit as exc:
            self.state = "sim"
            self.drone = None
            print(f"[SIM] Drone pairing exited ({exc}). Continuing without a drone.")
        except Exception as exc:
            self.state = "sim"
            self.drone = None
            print(f"[SIM] Drone pairing failed: {exc}")

    def takeoff(self):
        self.queue.put(("takeoff", None))

    def land(self):
        self.motion_token += 1
        self.current_motion = None
        self._clear_pending_commands()
        self.queue.put(("land", None))

    def stop(self):
        self.motion_token += 1
        self.current_motion = None
        self._clear_pending_commands()
        self.queue.put(("stop", None))

    def set_motion(self, motion: Optional[Tuple[str, ...]]):
        self.motion_token += 1
        token = self.motion_token
        if motion is None:
            self.current_motion = None
            self._clear_pending_motion_commands()
        self.queue.put(("motion", (token, motion)))

    def close(self):
        self.running = False
        self.queue.put(("close", None))
        self.worker.join(timeout=1.0)

    def _clear_pending_commands(self):
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break

    def _clear_pending_motion_commands(self):
        keep = []
        while True:
            try:
                item = self.queue.get_nowait()
            except Empty:
                break
            if item[0] != "motion":
                keep.append(item)
        for item in keep:
            self.queue.put(item)

    def _run(self):
        while self.running:
            try:
                kind, value = self.queue.get(timeout=COMMAND_TICK_S)
                while True:
                    try:
                        kind, value = self.queue.get_nowait()
                    except Empty:
                        break
                self._handle(kind, value)
            except Empty:
                self._send_motion_tick()

    def _handle(self, kind: str, value):
        if kind == "takeoff":
            self._takeoff()
        elif kind == "land":
            self._land()
        elif kind == "stop":
            self.current_motion = None
            self._stop_motion()
        elif kind == "motion":
            token, motion = value
            if token != self.motion_token:
                return
            value = motion
            if value is None:
                self.current_motion = None
                self._stop_motion()
                return
            if value != self.current_motion:
                self.current_motion = value
                self._send_motion_tick(force=True)

    def _takeoff(self):
        if self.state == "flying":
            return
        if self.drone is None:
            self.state = "flying"
            print("[SIM] takeoff")
            return
        try:
            self.drone.takeoff()
            self.state = "flying"
        except Exception as exc:
            print(f"[DRONE] takeoff failed: {exc}")

    def _land(self):
        self.current_motion = None
        self.state = "landing"
        self._stop_motion()
        if self.drone is None:
            self.state = "grounded"
            print("[SIM] land")
            return
        try:
            self.drone.land()
            self.state = "grounded"
        except Exception as exc:
            print(f"[DRONE] land failed: {exc}")

    def _stop_motion(self):
        if self.drone is None:
            self.state = "paused" if self.state == "flying" else self.state
            print("[SIM] stop")
            return

        self.last_step_at = 0.0
        neutral_sent = self._try_continuous_motion(None)

        for name in ("hover", "stop"):
            method = getattr(self.drone, name, None)
            if callable(method):
                try:
                    method()
                    if self.state == "flying":
                        self.state = "paused"
                    return
                except TypeError:
                    continue
                except Exception as exc:
                    print(f"[DRONE] {name} failed: {exc}")
                    return

        if not neutral_sent:
            print("[DRONE] no hover/stop method found; neutral command attempted")
        if self.state == "flying":
            self.state = "paused"

    def _send_motion_tick(self, force: bool = False):
        if self.current_motion is None or self.state not in ("flying", "paused"):
            return

        self.state = "flying"
        if self.drone is None:
            if force:
                print(f"[SIM] motion {_format_motion(self.current_motion)}")
            return

        if self._try_continuous_motion(self.current_motion):
            return

        now = time.monotonic()
        if force or now - self.last_step_at >= STEP_COOLDOWN_S:
            self.last_step_at = now
            self._fallback_step(self.current_motion)

    def _try_continuous_motion(self, motion: Optional[Tuple[str, ...]]) -> bool:
        if self.drone is None:
            return True

        setters = {
            "set_pitch": 0,
            "set_roll": 0,
            "set_throttle": 0,
            "set_yaw": 0,
        }
        motions = motion or ()
        if "Forward" in motions:
            setters["set_pitch"] = CONTROL_POWER
        elif "Back" in motions:
            setters["set_pitch"] = -CONTROL_POWER
        if "Right" in motions:
            setters["set_roll"] = CONTROL_POWER
        elif "Left" in motions:
            setters["set_roll"] = -CONTROL_POWER
        if "Up" in motions:
            setters["set_throttle"] = CONTROL_POWER
        elif "Down" in motions:
            setters["set_throttle"] = -CONTROL_POWER
        if "YawRight" in motions:
            setters["set_yaw"] = YAW_POWER
        elif "YawLeft" in motions:
            setters["set_yaw"] = -YAW_POWER
        unknown = [name for name in motions if name not in MOTION_NAMES]
        if unknown:
            return False

        if not all(callable(getattr(self.drone, name, None)) for name in setters):
            return False

        move = getattr(self.drone, "move", None)
        if not callable(move):
            return False

        try:
            for name, value in setters.items():
                getattr(self.drone, name)(value)
            try:
                move(COMMAND_TICK_S)
            except TypeError:
                move()
            return True
        except Exception as exc:
            print(f"[DRONE] continuous control failed, using step fallback: {exc}")
            return False

    def _fallback_step(self, motions: Tuple[str, ...]):
        vectors = {
            "Forward": (STEP_DISTANCE_M, 0, 0),
            "Back": (-STEP_DISTANCE_M, 0, 0),
            "Left": (0, STEP_DISTANCE_M, 0),
            "Right": (0, -STEP_DISTANCE_M, 0),
            "Up": (0, 0, STEP_DISTANCE_M),
            "Down": (0, 0, -STEP_DISTANCE_M),
        }
        if self.drone is None:
            return
        vec = [0.0, 0.0, 0.0]
        for motion in motions:
            delta = vectors.get(motion)
            if delta is None:
                continue
            vec[0] += delta[0]
            vec[1] += delta[1]
            vec[2] += delta[2]
        if vec == [0.0, 0.0, 0.0]:
            if "YawLeft" in motions:
                self._fallback_turn("left")
            elif "YawRight" in motions:
                self._fallback_turn("right")
            return
        try:
            self.drone.move_distance(vec[0], vec[1], vec[2], STEP_DURATION_S)
        except Exception as exc:
            print(f"[DRONE] step move failed: {exc}")

    def _fallback_turn(self, direction: str):
        method_names = ("turn_left", "rotate_left") if direction == "left" else ("turn_right", "rotate_right")
        for name in method_names:
            method = getattr(self.drone, name, None)
            if callable(method):
                try:
                    method(12)
                    return
                except TypeError:
                    try:
                        method()
                        return
                    except Exception as exc:
                        print(f"[DRONE] {name} failed: {exc}")
                        return
                except Exception as exc:
                    print(f"[DRONE] {name} failed: {exc}")
                    return
        print(f"[DRONE] no fallback turn method found for {direction}")


def build_layout(width: int, height: int):
    icon_h = max(70, int(height * 0.11))
    gap = 12
    left_w = int(width * 0.36)
    right_x = int(width * 0.58)
    mid_y1 = int(height * 0.36)
    mid_y2 = int(height * 0.64)
    bottom_y = height - gap
    top_y = icon_h + gap

    zones = [
        Rect("Up", gap, top_y, left_w, mid_y1 - gap, "Left", "^", COL_LEFT),
        Rect("Down", gap, mid_y2 + gap, left_w, bottom_y, "Left", "v", COL_LEFT),
        Rect("Forward", right_x, top_y, width - gap, mid_y1 - gap, "Right", "^", COL_RIGHT),
        Rect("Back", right_x, mid_y2 + gap, width - gap, bottom_y, "Right", "v", COL_RIGHT),
        Rect("Left", right_x, mid_y1, int((right_x + width) / 2) - gap, mid_y2, "Right", "<", COL_RIGHT),
        Rect("Right", int((right_x + width) / 2) + gap, mid_y1, width - gap, mid_y2, "Right", ">", COL_RIGHT),
    ]

    icon_w = int(width * 0.15)
    icons = [
        Rect("Takeoff", gap, gap, gap + icon_w, icon_h - gap, None, "TAKEOFF", (55, 170, 95)),
        Rect("Stop", int(width / 2 - icon_w / 2), gap, int(width / 2 + icon_w / 2), icon_h - gap, None, "STOP", (70, 145, 245)),
        Rect("Land", width - gap - icon_w, gap, width - gap, icon_h - gap, None, "LAND", COL_WARN),
    ]

    dead = Rect("Dead", left_w + gap, mid_y1, right_x - gap, mid_y2, None, "IDLE", COL_DEAD)
    return zones, icons, dead


def draw_rect(frame: np.ndarray, rect: Rect, active: bool, alpha: float = 0.32):
    color = COL_ACTIVE if active else rect.color
    if active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (rect.x1, rect.y1), (rect.x2, rect.y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, (rect.x1, rect.y1), (rect.x2, rect.y2), color, 2, cv2.LINE_AA)

    cx = (rect.x1 + rect.x2) // 2
    cy = (rect.y1 + rect.y2) // 2
    label = rect.symbol
    scale = 1.2 if len(label) <= 1 else 0.72
    thick = 3 if len(label) <= 1 else 2
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(frame, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, COL_TEXT, thick, cv2.LINE_AA)

    if rect.hand:
        cv2.putText(frame, rect.name, (rect.x1 + 12, rect.y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_TEXT, 2, cv2.LINE_AA)


def detect_icon(tips: List[Fingertip], icons: List[Rect]) -> Optional[str]:
    for icon_name in ("Land", "Stop", "Takeoff"):
        icon = next((r for r in icons if r.name == icon_name), None)
        if icon and any(icon.contains(tip.x, tip.y) for tip in tips):
            return icon_name
    return None


def _zone_priority(name: str) -> int:
    return MOTION_NAMES.index(name) if name in MOTION_NAMES else len(MOTION_NAMES)


def _format_motion(motions: Optional[Tuple[str, ...]]) -> str:
    if not motions:
        return "Idle"
    return " + ".join(MOTION_LABELS.get(motion, motion) for motion in motions)


def detect_motion_candidate(
    tips: List[Fingertip],
    zones: List[Rect],
    previous: Optional[Tuple[str, ...]],
) -> Optional[Tuple[str, ...]]:
    touched = []
    for zone in zones:
        for tip in tips:
            if tip.hand == zone.hand and zone.contains(tip.x, tip.y):
                touched.append(zone.name)
                break

    if not touched:
        return None

    left_hits = [name for name in touched if name in ("Up", "Down")]
    right_hits = [name for name in touched if name in ("Forward", "Back", "Left", "Right")]
    selected = []

    if len(left_hits) == 1:
        selected.append(left_hits[0])
    elif previous:
        previous_left = [name for name in previous if name in left_hits]
        if previous_left:
            selected.append(previous_left[0])

    if len(right_hits) == 1:
        selected.append(right_hits[0])
    elif previous:
        previous_right = [name for name in previous if name in right_hits]
        if previous_right:
            selected.append(previous_right[0])

    if not selected:
        return None
    return tuple(sorted(selected, key=_zone_priority))


def detect_hovered_zones(tips: List[Fingertip], zones: List[Rect]) -> Tuple[str, ...]:
    hovered = []
    for zone in zones:
        if any(tip.hand == zone.hand and zone.contains(tip.x, tip.y) for tip in tips):
            hovered.append(zone.name)
    return tuple(sorted(hovered, key=_zone_priority))


def extract_fingertips(result, width: int, height: int) -> List[Fingertip]:
    tips = []
    if not result.hand_landmarks:
        return tips

    for hand_index, hand in enumerate(result.hand_landmarks):
        hand_label = "Unknown"
        if result.handedness and hand_index < len(result.handedness) and result.handedness[hand_index]:
            hand_label = result.handedness[hand_index][0].category_name

        if hand_label not in ("Left", "Right"):
            continue

        # Detection runs on the mirrored camera frame, so swap MediaPipe's label
        # back to the user's real hand for the on-screen control mapping.
        hand_label = "Right" if hand_label == "Left" else "Left"

        for idx in (THUMB_TIP_ID, INDEX_TIP_ID):
            lm = hand[idx]
            x = int(np.clip(lm.x * width, 0, width - 1))
            y = int(np.clip(lm.y * height, 0, height - 1))
            tips.append(Fingertip(hand_label, idx, x, y))

    return tips


def index_tips_only(tips: List[Fingertip]) -> List[Fingertip]:
    return [tip for tip in tips if tip.idx == INDEX_TIP_ID]


def draw_overlay(
    frame: np.ndarray,
    zones: List[Rect],
    icons: List[Rect],
    dead: Rect,
    tips: List[Fingertip],
    active_motion: Optional[Tuple[str, ...]],
    hovered_motion: Tuple[str, ...],
    pinch: Optional[PinchInfo],
    active_icon: Optional[str],
    icon_progress: float,
    drone_state: str,
):
    H, W = frame.shape[:2]

    for zone in zones:
        draw_rect(frame, zone, zone.name in (active_motion or ()) or zone.name in hovered_motion)

    for icon in icons:
        draw_rect(frame, icon, icon.name == active_icon, alpha=0.42)
        if icon.name == active_icon and icon_progress > 0:
            fill_w = int((icon.x2 - icon.x1) * icon_progress)
            cv2.rectangle(frame, (icon.x1, icon.y2 - 6), (icon.x1 + fill_w, icon.y2), COL_ACTIVE, -1)

    for tip in index_tips_only(tips):
        color = COL_LEFT_TIP if tip.hand == "Left" else COL_RIGHT_TIP
        cv2.circle(frame, (tip.x, tip.y), 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (tip.x, tip.y), 9, COL_TEXT, 1, cv2.LINE_AA)

    if pinch:
        line_color = COL_ACTIVE if pinch.motion else (255, 220, 120)
        cv2.arrowedLine(frame, pinch.anchor, pinch.mid, line_color, 4, cv2.LINE_AA, tipLength=0.18)
        cv2.circle(frame, pinch.anchor, 8, COL_TEXT, 2, cv2.LINE_AA)
        cv2.circle(frame, pinch.mid, 10, COL_TEXT, 2, cv2.LINE_AA)
        direction = MOTION_LABELS.get(pinch.motion, "Pinch")
        cv2.putText(frame, f"{direction}  dx={pinch.drag_dx}", (pinch.mid[0] + 14, pinch.mid[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 2, cv2.LINE_AA)

    command = _format_motion(active_motion)
    hover = _format_motion(hovered_motion) if hovered_motion else active_icon or ("Pinch" if pinch else "None")
    cv2.putText(frame, f"Command: {command}", (18, H - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, COL_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Hover: {hover}", (18, H - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COL_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Drone: {drone_state}", (18, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COL_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, "Index zones  Pinch+drag=rotate  T=takeoff  Space=stop  L=land  Q/Esc=quit", (W - 690, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_DIM, 1, cv2.LINE_AA)


def main():
    controller = DroneController()
    controller.pair()

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        controller.close()
        exit(1)

    cv2.namedWindow(WIN_NAME)
    smoother = FingertipSmoother(TIP_SMOOTHING)
    pinch_tracker = PinchTracker()
    motion_debouncer = ZoneDebouncer(ZONE_DEBOUNCE_S)
    icon_dwell = IconDwell(ICON_DWELL_S)
    last_requested_motion = object()
    last_hover_feedback = None
    should_exit = False
    start_time = time.monotonic()

    try:
        while not should_exit:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("[WARNING] Empty frame. Skipping...")
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("t"):
                controller.takeoff()
            elif key == ord("l"):
                controller.land()
            elif key == ord(" "):
                controller.stop()

            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            zones, icons, dead = build_layout(W, H)

            ts_ms = int((time.monotonic() - start_time) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            raw_tips = extract_fingertips(result, W, H)
            tips = smoother.update(raw_tips)
            control_tips = index_tips_only(tips)
            pinch, pinch_motion = pinch_tracker.update(tips)
            now = time.monotonic()

            active_icon = detect_icon(control_tips, icons)
            fired_icon, icon_progress = icon_dwell.update(active_icon, now)
            if fired_icon == "Takeoff":
                print("[ICON] Takeoff dwell complete")
                controller.takeoff()
            elif fired_icon == "Land":
                print("[ICON] Land dwell complete")
                controller.land()
            elif fired_icon == "Stop":
                print("[ICON] Stop dwell complete - exiting")
                controller.stop()
                should_exit = True
                continue

            zone_controls_enabled = active_icon is None and pinch is None
            hovered_motion = () if not zone_controls_enabled else detect_hovered_zones(control_tips, zones)
            candidate = None if not zone_controls_enabled else detect_motion_candidate(control_tips, zones, motion_debouncer.active)
            active_motion = motion_debouncer.update(candidate, now)
            command_motion = pinch_motion if active_icon is None and pinch else active_motion

            feedback = active_icon or pinch_motion or ("Pinch" if pinch else None) or hovered_motion
            if feedback != last_hover_feedback:
                if feedback:
                    if isinstance(feedback, tuple):
                        print(f"[HOVER] {_format_motion(feedback)}")
                    else:
                        print(f"[HOVER] {feedback}")
                else:
                    print("[HOVER] Idle")
                last_hover_feedback = feedback

            requested_motion = None if active_icon else command_motion
            if requested_motion != last_requested_motion:
                print(f"[COMMAND] {_format_motion(requested_motion)}")
                controller.set_motion(requested_motion)
                last_requested_motion = requested_motion

            draw_overlay(frame, zones, icons, dead, tips, command_motion, hovered_motion, pinch, active_icon, icon_progress, controller.state)
            cv2.imshow(WIN_NAME, frame)
    finally:
        controller.stop()
        controller.close()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
