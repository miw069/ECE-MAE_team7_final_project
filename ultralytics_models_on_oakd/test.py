import os

# Suppress Qt font-directory warnings before cv2 is imported.
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

import pathlib
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import depthai as dai
import numpy as np
import ultralytics
from depthai_nodes.node import ParsingNeuralNetwork
from ultralytics.cfg import get_cfg
from ultralytics.trackers.bot_sort import BOTSORT

MODEL_PATH = pathlib.Path(__file__).with_name("yolo26n.rvc2.tar.xz")
FPS_LIMIT = 10
PERSON_CLASS = 0
LOST_TIMEOUT_FRAMES = 30
INFO_PANEL_HEIGHT = 92
WINDOW_NAME = "Person Tracker"

COLOR_NORMAL = (0, 200, 0)
COLOR_LOCKED = (0, 100, 255)
COLOR_LOST = (0, 255, 255)
COLOR_PANEL = (28, 28, 28)
COLOR_TEXT = (230, 230, 230)
COLOR_SUBTEXT = (150, 150, 150)

EXTENDED_DISPARITY = False
SUBPIXEL = False
LR_CHECK = True


@dataclass
class TrackedPerson:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class TrackingState(Enum):
    ARMED = "ARMED"
    LOCKED = "LOCKED"
    LOST = "LOST"


class TargetStateMachine:
    def __init__(self, lost_timeout: int = LOST_TIMEOUT_FRAMES):
        self.state = TrackingState.ARMED
        self.target_id: int | None = None
        self.target_bbox: tuple[int, int, int, int] | None = None
        self.lost_frames = 0
        self.lost_timeout = lost_timeout

    def select_target(self, track_id: int, persons: list[TrackedPerson]) -> bool:
        target = next((p for p in persons if p.track_id == track_id), None)
        if target is None:
            return False
        self.state = TrackingState.LOCKED
        self.target_id = track_id
        self.target_bbox = target.bbox
        self.lost_frames = 0
        return True

    def cancel(self) -> None:
        self.state = TrackingState.ARMED
        self.target_id = None
        self.target_bbox = None
        self.lost_frames = 0

    def update(self, persons: list[TrackedPerson]) -> None:
        if self.state == TrackingState.ARMED:
            return

        target = next((p for p in persons if p.track_id == self.target_id), None)
        if target is not None:
            self.state = TrackingState.LOCKED
            self.target_bbox = target.bbox
            self.lost_frames = 0
            return

        self.lost_frames += 1
        if self.target_bbox is not None:
            self.state = TrackingState.LOST

        if self.lost_frames >= self.lost_timeout:
            self.cancel()


class MouseState:
    def __init__(self) -> None:
        self.pending_click: tuple[int, int] | None = None
        self.display_size = (0, 0)
        self.tracking_size = (0, 0)

    def callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and y < self.display_size[1]:
            self.pending_click = (x, y)

    def consume_tracking_click(self) -> tuple[int, int] | None:
        if self.pending_click is None:
            return None
        click_x, click_y = self.pending_click
        self.pending_click = None

        display_w, display_h = self.display_size
        tracking_w, tracking_h = self.tracking_size
        if min(display_w, display_h, tracking_w, tracking_h) <= 0:
            return None

        mapped_x = int(click_x * tracking_w / display_w)
        mapped_y = int(click_y * tracking_h / display_h)
        return mapped_x, mapped_y


class _Boxes:
    """Duck-type the ultralytics Boxes interface using numpy arrays."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).ravel()
        self.cls = np.asarray(cls, dtype=np.float32).ravel()
        if len(self.xyxy) == 0:
            self.xywh = np.empty((0, 4), dtype=np.float32)
            return
        xc = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2
        yc = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2
        bw = self.xyxy[:, 2] - self.xyxy[:, 0]
        bh = self.xyxy[:, 3] - self.xyxy[:, 1]
        self.xywh = np.stack([xc, yc, bw, bh], axis=1)

    def __len__(self):
        return len(self.xyxy)

    def __getitem__(self, idx):
        return _Boxes(self.xyxy[idx], self.conf[idx], self.cls[idx])


def _select_stereo_preset():
    preset_mode = dai.node.StereoDepth.PresetMode
    for name in ("HIGH_DENSITY", "HIGH_DETAIL", "DEFAULT"):
        preset = getattr(preset_mode, name, None)
        if preset is not None:
            return preset, name
    raise AttributeError("No supported StereoDepth preset found in depthai build")


def _create_pipeline(device: dai.Device):
    pipeline = dai.Pipeline(device)

    cam_rgb = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    model_archive = dai.NNArchive(archivePath=str(MODEL_PATH))
    nn_node = pipeline.create(ParsingNeuralNetwork).build(
        cam_rgb, model_archive, fps=FPS_LIMIT
    )

    video_q = nn_node.passthrough.createOutputQueue()
    det_q = nn_node.out.createOutputQueue()

    disparity_q = None
    max_disparity = 1.0

    try:
        stereo_preset, stereo_name = _select_stereo_preset()
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        stereo.setDefaultProfilePreset(stereo_preset)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(LR_CHECK)
        stereo.setExtendedDisparity(EXTENDED_DISPARITY)
        stereo.setSubpixel(SUBPIXEL)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        max_disparity = max(float(stereo.initialConfig.getMaxDisparity()), 1.0)
        disparity_q = stereo.disparity.createOutputQueue()
        print(f"[Depth] Disparity enabled with preset {stereo_name}.")
    except Exception as exc:
        print(f"[Depth] Disparity unavailable: {exc}")

    return pipeline, video_q, det_q, disparity_q, max_disparity


def _get_colorized_disparity(disparity_q, max_disparity: float) -> np.ndarray | None:
    if disparity_q is None:
        return None
    msg = disparity_q.tryGet()
    if msg is None:
        return None
    frame = msg.getFrame()
    normalized = (frame * (255.0 / max_disparity)).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def _tracks_to_persons(tracks) -> list[TrackedPerson]:
    persons: list[TrackedPerson] = []
    for track in tracks:
        x1, y1, x2, y2 = (int(track[0]), int(track[1]), int(track[2]), int(track[3]))
        persons.append(
            TrackedPerson(
                track_id=int(track[4]),
                bbox=(x1, y1, x2, y2),
                confidence=float(track[5]),
            )
        )
    return persons


def _find_nearest_person(px: int, py: int, persons: list[TrackedPerson]) -> TrackedPerson | None:
    for person in persons:
        x1, y1, x2, y2 = person.bbox
        if x1 <= px <= x2 and y1 <= py <= y2:
            return person
    if not persons:
        return None
    return min(persons, key=lambda p: (p.center[0] - px) ** 2 + (p.center[1] - py) ** 2)


def _draw_dashed_rect(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
) -> None:
    x1, y1, x2, y2 = bbox
    edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (sx, sy), (ex, ey) in edges:
        dist = int(np.hypot(ex - sx, ey - sy))
        for i in range(0, dist, dash_len * 2):
            t0 = i / max(dist, 1)
            t1 = min((i + dash_len) / max(dist, 1), 1.0)
            p0 = (int(sx + (ex - sx) * t0), int(sy + (ey - sy) * t0))
            p1 = (int(sx + (ex - sx) * t1), int(sy + (ey - sy) * t1))
            cv2.line(image, p0, p1, color, thickness)


def _draw_tracking_frame(
    frame: np.ndarray,
    persons: list[TrackedPerson],
    state_machine: TargetStateMachine,
) -> np.ndarray:
    canvas = frame.copy()
    for person in persons:
        is_target = person.track_id == state_machine.target_id
        color = COLOR_NORMAL
        thickness = 2
        if is_target:
            color = COLOR_LOST if state_machine.state == TrackingState.LOST else COLOR_LOCKED
            thickness = 3

        x1, y1, x2, y2 = person.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"ID {person.track_id}  {person.confidence:.0%}"
        cv2.putText(
            canvas,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if state_machine.state == TrackingState.LOST and state_machine.target_bbox is not None:
        _draw_dashed_rect(canvas, state_machine.target_bbox, COLOR_LOST, thickness=2)

    return canvas


def _render_info_panel(
    frame: np.ndarray,
    state_machine: TargetStateMachine,
    fps: float,
    view_mode: str,
) -> np.ndarray:
    panel = np.full((INFO_PANEL_HEIGHT, frame.shape[1], 3), COLOR_PANEL, dtype=np.uint8)

    if state_machine.state == TrackingState.ARMED:
        status_text = "Click a person to lock on."
        status_color = (200, 200, 0)
    elif state_machine.state == TrackingState.LOCKED:
        status_text = f"Locked on ID {state_machine.target_id}"
        status_color = (0, 180, 0)
    else:
        status_text = (
            f"Lost target ID {state_machine.target_id} - reacquiring "
            f"({state_machine.lost_frames}/{state_machine.lost_timeout})"
        )
        status_color = (0, 180, 255)

    cv2.putText(
        panel,
        f"State: {state_machine.state.value}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"{fps:.1f} fps",
        (frame.shape[1] - 90, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"View: {'Disparity' if view_mode == 'disparity' else 'Tracking'}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        status_text,
        (150, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "[Click] Select  [C/Esc] Cancel  [V] Toggle view  [Q] Quit",
        (10, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_SUBTEXT,
        1,
        cv2.LINE_AA,
    )

    return np.vstack([frame, panel])


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model archive not found: {MODEL_PATH}")

    tracker_cfg = str(pathlib.Path(ultralytics.__file__).parent / "cfg/trackers/botsort.yaml")
    tracker = BOTSORT(get_cfg(tracker_cfg), frame_rate=FPS_LIMIT)
    state_machine = TargetStateMachine()
    mouse_state = MouseState()
    view_mode = "tracking"
    latest_detections = []
    last_disparity_frame = None
    prev_time = time.perf_counter()

    device = dai.Device()
    print(f"[DepthAI] Platform: {device.getPlatformAsString()}")

    pipeline, video_q, det_q, disparity_q, max_disparity = _create_pipeline(device)
    with pipeline:
        pipeline.start()

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, mouse_state.callback)
        print("[UI] Click a person to lock on. Press C or Esc to cancel, V to toggle view, Q to quit.")

        while pipeline.isRunning():
            frame_msg = video_q.tryGet()
            det_msg = det_q.tryGet()

            if det_msg is not None:
                latest_detections = [
                    detection
                    for detection in det_msg.detections
                    if detection.label == PERSON_CLASS
                ]

            if disparity_q is not None:
                disparity_frame = _get_colorized_disparity(disparity_q, max_disparity)
                if disparity_frame is not None:
                    last_disparity_frame = disparity_frame

            if frame_msg is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in (ord("c"), 27):
                    state_machine.cancel()
                elif key == ord("v"):
                    view_mode = "disparity" if view_mode == "tracking" else "tracking"
                continue

            frame = frame_msg.getCvFrame()
            mouse_state.tracking_size = (frame.shape[1], frame.shape[0])
            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            boxes = _Boxes(
                [[d.xmin * frame.shape[1], d.ymin * frame.shape[0], d.xmax * frame.shape[1], d.ymax * frame.shape[0]]
                 for d in latest_detections],
                [d.confidence for d in latest_detections],
                [PERSON_CLASS] * len(latest_detections),
            )
            persons = _tracks_to_persons(tracker.update(boxes, frame))

            mapped_click = mouse_state.consume_tracking_click()
            if mapped_click is not None:
                clicked_person = _find_nearest_person(
                    mapped_click[0],
                    mapped_click[1],
                    persons,
                )
                if clicked_person is not None:
                    state_machine.select_target(clicked_person.track_id, persons)

            state_machine.update(persons)

            tracking_frame = _draw_tracking_frame(frame, persons, state_machine)
            if view_mode == "disparity" and last_disparity_frame is not None:
                display = last_disparity_frame
            else:
                display = tracking_frame
            mouse_state.display_size = (display.shape[1], display.shape[0])
            display = _render_info_panel(display, state_machine, fps, view_mode)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("c"), 27):
                state_machine.cancel()
            elif key == ord("v"):
                view_mode = "disparity" if view_mode == "tracking" else "tracking"

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()