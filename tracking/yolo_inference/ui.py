"""Frame annotation for the person tracking system.

FrameAnnotator – pure rendering, no window or keyboard code.
TrackingUI     – extends FrameAnnotator with legacy OpenCV window support
                 so that main.py continues to work unchanged.
"""

import time
from typing import List, Optional

import cv2
import numpy as np

from detector import TrackedPerson
from state_machine import TargetStateMachine, TrackingState
import config


class FrameAnnotator:
    """Draws person overlays on video frames. No window management."""

    def render(
        self,
        frame: np.ndarray,
        persons: List[TrackedPerson],
        sm: TargetStateMachine,
    ) -> np.ndarray:
        """Draw all overlays and return the annotated frame."""
        canvas = frame.copy()

        for person in persons:
            is_target = (person.track_id == sm.target_id)
            self._draw_person(canvas, person, is_target, sm.state)

        # If LOST, draw last-known bbox as dashed
        if sm.state == TrackingState.LOST and sm.target_bbox is not None:
            self._draw_dashed_rect(canvas, sm.target_bbox, config.COLOR_LOST, thickness=2)

        return canvas

    # ── private helpers ────────────────────────────────────

    def _draw_person(
        self,
        img: np.ndarray,
        person: TrackedPerson,
        is_target: bool,
        state: TrackingState,
    ) -> None:
        x1, y1, x2, y2 = person.bbox

        if is_target:
            color = config.COLOR_LOST if state == TrackingState.LOST else config.COLOR_LOCKED
            thickness = config.BOX_THICKNESS_TARGET
        else:
            color = config.COLOR_NORMAL
            thickness = config.BOX_THICKNESS_NORMAL

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = f"ID:{person.track_id}  {person.confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = config.FONT_SCALE
        ft = config.FONT_THICKNESS

        (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, cv2.FILLED)
        cv2.putText(img, label, (x1 + 3, y1 - 5), font, fs, (0, 0, 0), ft, cv2.LINE_AA)

    @staticmethod
    def _draw_dashed_rect(
        img: np.ndarray, bbox: tuple, color: tuple, thickness: int = 2, dash_len: int = 10
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
                cv2.line(img, p0, p1, color, thickness)


class TrackingUI(FrameAnnotator):
    """Legacy OpenCV window UI.  Extends FrameAnnotator with keyboard input
    and an info panel so that main.py works without modification."""

    WINDOW_NAME = "Person Tracker"

    def __init__(self):
        self.input_buffer: str = ""
        self.fps: float = 0.0
        self._prev_time: float = time.time()
        self._flash_msg: str = ""
        self._flash_until: float = 0.0

    # ── rendering override ──────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        persons: List[TrackedPerson],
        sm: TargetStateMachine,
    ) -> np.ndarray:
        """Draw overlays + info panel."""
        self._update_fps()
        canvas = super().render(frame, persons, sm)
        canvas = self._draw_info_panel(canvas, sm)
        return canvas

    # ── input handling ─────────────────────────────────────

    def process_key(self, key: int) -> Optional[str]:
        if key == -1:
            return None

        char = chr(key & 0xFF)

        if char in ("q", "Q") or key == 27:
            return "quit"

        if char in ("c", "C"):
            self.input_buffer = ""
            return "cancel"

        if char.isdigit():
            self.input_buffer += char
            return None

        if key in (8, 127):
            self.input_buffer = self.input_buffer[:-1]
            return None

        if key in (13, 10):
            if self.input_buffer:
                track_id = self.input_buffer
                self.input_buffer = ""
                return f"select:{track_id}"
            return None

        return None

    def flash(self, msg: str, duration: float = 2.0) -> None:
        self._flash_msg = msg
        self._flash_until = time.time() + duration

    # ── private helpers ────────────────────────────────────

    def _update_fps(self) -> None:
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        self.fps = 1.0 / max(dt, 1e-6)

    def _draw_info_panel(self, frame: np.ndarray, sm: TargetStateMachine) -> np.ndarray:
        h, w = frame.shape[:2]
        panel_h = config.INFO_PANEL_HEIGHT
        panel = np.full((panel_h, w, 3), config.COLOR_TEXT_BG, dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.55
        ft = 1
        white = config.COLOR_WHITE
        y_line = 22

        state_color = {
            TrackingState.ARMED:  config.COLOR_STATE_ARMED,
            TrackingState.LOCKED: config.COLOR_STATE_LOCKED,
            TrackingState.LOST:   config.COLOR_STATE_LOST,
        }.get(sm.state, white)

        state_txt = f"State: {sm.state.value}"
        if sm.target_id is not None:
            state_txt += f"   Target ID: {sm.target_id}"
        if sm.state == TrackingState.LOST:
            state_txt += f"   (lost {sm.lost_frames}/{sm.lost_timeout} frames)"

        cv2.putText(panel, state_txt, (10, y_line), font, fs, state_color, ft, cv2.LINE_AA)
        fps_txt = f"FPS: {self.fps:.1f}"
        cv2.putText(panel, fps_txt, (w - 120, y_line), font, fs, white, ft, cv2.LINE_AA)

        y_line += 28
        input_txt = f"Select ID: {self.input_buffer}_" if self.input_buffer else "Select ID: _"
        cv2.putText(panel, input_txt, (10, y_line), font, fs, (180, 220, 255), ft, cv2.LINE_AA)

        if self._flash_msg and time.time() < self._flash_until:
            cv2.putText(panel, self._flash_msg, (220, y_line), font, fs, (100, 180, 255), ft, cv2.LINE_AA)

        y_line += 28
        help_txt = "[0-9] Type ID  |  [Enter] Lock  |  [C] Cancel  |  [Backspace] Delete  |  [Q/Esc] Quit"
        cv2.putText(panel, help_txt, (10, y_line), font, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

        return np.vstack([frame, panel])
