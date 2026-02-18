#!/usr/bin/env python3
"""FastAPI web server for the person tracking system.

Run:
    uv run uvicorn server:app --host 0.0.0.0 --port 8000

Or directly:
    python server.py
"""

import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from typing import AsyncGenerator

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

import config
import depth_overlay as _depth_overlay
from camera import OakDLiteCamera
from detector import PersonDetector
from frame_store import FrameStore, TrackingSnapshot
from reid_manager import ReIDManager
from state_machine import TargetStateMachine, TrackingState
from ui import FrameAnnotator

# ── Pydantic models ──────────────────────────────────────
from pydantic import BaseModel


class ClickPayload(BaseModel):
    x_norm: float
    y_norm: float


# ── App + global store ───────────────────────────────────

app = FastAPI(title="Person Tracker")
store = FrameStore()

# Thread-safe toggle for depth-perception overlay
depth_toggle = threading.Event()   # set = enabled, cleared = disabled

STATIC_DIR = Path(__file__).parent / "static"


# ── Camera loop (background thread) ─────────────────────

def camera_loop(store: FrameStore) -> None:
    """Initialise camera/detector/state-machine and feed frames into FrameStore."""
    try:
        cam = OakDLiteCamera(
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=config.CAMERA_FPS,
        )
    except Exception as exc:
        print(f"[CameraLoop] Failed to open camera: {exc}")
        print("[CameraLoop] Falling back to synthetic frames for testing.")
        _synthetic_loop(store)
        return

    reid = ReIDManager()
    detector = PersonDetector()
    sm = TargetStateMachine(reid_manager=reid)
    annotator = FrameAnnotator()

    print("[CameraLoop] Starting capture loop.")
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.005)
                continue

            # Detect + track
            persons = detector.update(frame)

            # Handle pending UI actions
            action = store.consume_click()
            if action is not None:
                if action[0] == "cancel":
                    sm.cancel()
                else:
                    x_norm, y_norm = action
                    h, w = frame.shape[:2]
                    px, py = x_norm * w, y_norm * h
                    # Find the person whose bounding-box centre is closest to the click
                    best_id = _find_nearest_person(px, py, persons)
                    if best_id is not None:
                        sm.select_target(best_id, persons)

            # Update state machine (pass frame for Re-ID embedding)
            sm.update(persons, frame=frame)

            # Render annotations
            annotated = annotator.render(frame, persons, sm)

            # Depth-perception overlay (toggled via /api/depth-toggle)
            if depth_toggle.is_set():
                depth_frame = cam.get_depth_frame()
                if depth_frame is not None:
                    _depth_overlay.draw_overlay(annotated, depth_frame)

            # Encode to JPEG
            ok, buf = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
            )
            if not ok:
                continue

            snapshot = TrackingSnapshot(
                state=sm.state.value,
                target_id=sm.target_id,
                lost_frames=sm.lost_frames,
                lost_timeout=sm.lost_timeout,
                fps=0.0,  # FrameStore fills this in
            )
            store.put_frame(buf.tobytes(), snapshot)

    except Exception as exc:
        print(f"[CameraLoop] Error: {exc}")
    finally:
        cam.release()
        print("[CameraLoop] Camera released.")


def _find_nearest_person(px: float, py: float, persons) -> int | None:
    """Return the track_id of the person whose bbox contains (px,py),
    or the nearest centre if none contains it."""
    # First check containment
    for p in persons:
        x1, y1, x2, y2 = p.bbox
        if x1 <= px <= x2 and y1 <= py <= y2:
            return p.track_id
    # Fall back to nearest centre
    if not persons:
        return None
    best = min(persons, key=lambda p: (p.center[0] - px) ** 2 + (p.center[1] - py) ** 2)
    return best.track_id


def _synthetic_loop(store: FrameStore) -> None:
    """Emit a solid-colour frame so MJPEG/SSE endpoints can be tested without camera."""
    import numpy as np

    frame = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        frame, "No camera – synthetic feed",
        (30, config.CAMERA_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 100), 2, cv2.LINE_AA,
    )

    sm_dummy = TargetStateMachine()  # stays ARMED
    snapshot = TrackingSnapshot(
        state=sm_dummy.state.value,
        target_id=None,
        lost_frames=0,
        lost_timeout=sm_dummy.lost_timeout,
        fps=0.0,
    )

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    jpeg = buf.tobytes() if ok else b""

    while True:
        store.put_frame(jpeg, snapshot)
        time.sleep(1.0 / config.CAMERA_FPS)


# ── Startup ──────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    loop = asyncio.get_event_loop()
    store.set_loop(loop)
    t = threading.Thread(target=camera_loop, args=(store,), daemon=True)
    t.start()
    print(f"[Server] Camera thread started. Serving on http://{config.SERVER_HOST}:{config.SERVER_PORT}")


# ── Endpoints ────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/stream")
async def stream() -> StreamingResponse:
    async def mjpeg_generator() -> AsyncGenerator[bytes, None]:
        boundary = b"--frame"
        while True:
            jpeg = store.get_frame()
            if jpeg:
                yield (
                    boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.016)  # ~60 Hz poll

    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
async def status_sse() -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    store.subscribe_sse(queue)

    async def event_generator() -> AsyncGenerator[bytes, None]:
        try:
            while True:
                payload = await queue.get()
                yield f"data: {payload}\n\n".encode()
        except asyncio.CancelledError:
            pass
        finally:
            store.unsubscribe_sse(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/select")
async def api_select(payload: ClickPayload) -> dict:
    store.post_click(payload.x_norm, payload.y_norm)
    return {"status": "ok"}


@app.post("/api/cancel")
async def api_cancel() -> dict:
    store.post_cancel()
    return {"status": "ok"}


@app.post("/api/depth-toggle")
async def api_depth_toggle() -> dict:
    if depth_toggle.is_set():
        depth_toggle.clear()
    else:
        depth_toggle.set()
    return {"status": "ok", "depth_enabled": depth_toggle.is_set()}


# ── Direct execution ─────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=False,
    )
