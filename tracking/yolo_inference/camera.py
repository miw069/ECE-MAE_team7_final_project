"""OAK-D Lite camera capture using DepthAI 3.x."""

import depthai as dai
import numpy as np

import depth_overlay


class OakDLiteCamera:
    """Captures BGR frames from an OAK-D Lite camera via DepthAI 3.x pipeline."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

        self._device = dai.Device()
        platform = self._device.getPlatform().name
        frame_type = (
            dai.ImgFrame.Type.BGR888p if platform == "RVC2"
            else dai.ImgFrame.Type.BGR888i
        )

        self._pipeline = dai.Pipeline(self._device)
        cam = self._pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput((width, height), frame_type, fps=fps)
        self._queue = cam_out.createOutputQueue()

        # Extend pipeline with stereo depth (optional – fails gracefully)
        self._depth_queue = None
        depth_out = depth_overlay.setup_stereo(self._pipeline)
        if depth_out is not None:
            try:
                self._depth_queue = depth_out.createOutputQueue()
                print("[Camera] Depth overlay queue ready.")
            except Exception as exc:
                print(f"[Camera] Depth queue creation failed: {exc}")

        self._pipeline.start()
        print(f"[Camera] OAK-D Lite opened – {width}x{height} @ {fps} fps ({platform})")

    def get_depth_frame(self) -> "np.ndarray | None":
        """Return latest stereo depth frame (uint16 mm values), or None."""
        if self._depth_queue is None:
            return None
        try:
            msg = self._depth_queue.tryGet()
            return msg.getCvFrame() if msg is not None else None
        except Exception:
            return None

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read a frame. Returns (success, frame) matching cv2.VideoCapture API."""
        in_frame = self._queue.tryGet()
        if in_frame is None:
            return False, None
        return True, in_frame.getCvFrame()

    def release(self) -> None:
        """Stop the pipeline and close the device."""
        self._pipeline.stop()
        self._device.close()
        print("[Camera] OAK-D Lite closed.")
