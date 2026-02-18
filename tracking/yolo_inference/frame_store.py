"""Thread-safe hub between the camera thread and FastAPI handlers.

The camera thread writes frames and state snapshots; FastAPI handlers
read frames, subscribe to SSE events, and post click/cancel actions.
"""

import asyncio
import json
import threading
import time
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class TrackingSnapshot:
    """Serialisable snapshot of the tracking state for SSE."""
    state: str          # "ARMED" | "LOCKED" | "LOST"
    target_id: Optional[int]
    lost_frames: int
    lost_timeout: int
    fps: float


class FrameStore:
    def __init__(self):
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None  # JPEG bytes

        self._click_lock = threading.Lock()
        self._pending_click: Optional[tuple] = None   # (x_norm, y_norm) or None
        self._pending_cancel: bool = False

        self._sse_lock = threading.Lock()
        self._sse_queues: List[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._fps_lock = threading.Lock()
        self._frame_times: List[float] = []
        self._fps: float = 0.0

    # ── event loop registration ────────────────────────────

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the asyncio event loop (called from startup handler)."""
        self._loop = loop

    # ── frame writer (camera thread) ───────────────────────

    def put_frame(self, jpeg_bytes: bytes, snapshot: TrackingSnapshot) -> None:
        """Store the latest JPEG and broadcast a state SSE event."""
        with self._frame_lock:
            self._latest_frame = jpeg_bytes

        # Update FPS counter
        now = time.monotonic()
        with self._fps_lock:
            self._frame_times.append(now)
            cutoff = now - 1.0
            self._frame_times = [t for t in self._frame_times if t >= cutoff]
            self._fps = float(len(self._frame_times))

        snapshot.fps = self._fps

        self._broadcast_sse(snapshot)

    # ── frame reader (MJPEG endpoint) ─────────────────────

    def get_frame(self) -> Optional[bytes]:
        """Return the latest JPEG bytes, or None if no frame yet."""
        with self._frame_lock:
            return self._latest_frame

    # ── click/cancel writers (REST endpoints) ─────────────

    def post_click(self, x_norm: float, y_norm: float) -> None:
        with self._click_lock:
            self._pending_click = (x_norm, y_norm)

    def post_cancel(self) -> None:
        with self._click_lock:
            self._pending_cancel = True

    # ── click/cancel reader (camera thread) ───────────────

    def consume_click(self) -> Optional[tuple]:
        """Pop and return a pending click as (x_norm, y_norm) or ("cancel",) or None."""
        with self._click_lock:
            if self._pending_cancel:
                self._pending_cancel = False
                self._pending_click = None
                return ("cancel",)
            if self._pending_click is not None:
                click = self._pending_click
                self._pending_click = None
                return click
        return None

    # ── SSE subscriber management ─────────────────────────

    def subscribe_sse(self, queue: asyncio.Queue) -> None:
        with self._sse_lock:
            self._sse_queues.append(queue)

    def unsubscribe_sse(self, queue: asyncio.Queue) -> None:
        with self._sse_lock:
            try:
                self._sse_queues.remove(queue)
            except ValueError:
                pass

    def _broadcast_sse(self, snapshot: TrackingSnapshot) -> None:
        if self._loop is None:
            return
        payload = json.dumps(asdict(snapshot))
        with self._sse_lock:
            queues = list(self._sse_queues)
        for q in queues:
            asyncio.run_coroutine_threadsafe(q.put(payload), self._loop)
