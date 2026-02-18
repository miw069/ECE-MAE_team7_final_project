"""Depth-perception overlay using OAK-D Lite stereo depth.

Two public functions:
  setup_stereo(pipeline)         – append stereo nodes to an existing pipeline;
                                   returns the depth output socket (or None on failure).
  draw_overlay(frame, depth_np)  – draw distance-coloured grid cells onto a frame in-place.

The overlay is computed directly from the raw depth frame (mm values) instead of
the SpatialLocationCalculator, which has broken spatial-coordinate output with a
manually configured DAI 3.x pipeline.
"""

import numpy as np
import cv2
import depthai as dai

# Distance thresholds in mm
WARNING_MM = 1000   # 1 m  → orange
CRITICAL_MM = 500   # 50 cm → red

# Grid dimensions (columns × rows)
GRID_X = 15
GRID_Y = 9

# Percentile used per cell — low value catches the closest valid pixels
# (10th percentile ≈ "what's nearest in this zone", ignoring noise)
_DEPTH_PERCENTILE = 10

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def setup_stereo(pipeline: dai.Pipeline):
    """Append left/right MonoCamera + StereoDepth nodes to *pipeline*.

    Must be called **before** ``pipeline.start()``.

    Returns the stereo ``depth`` output socket on success (call
    ``.createOutputQueue()`` on it), or ``None`` if setup fails.
    """
    try:
        left = pipeline.create(dai.node.MonoCamera)
        left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        right = pipeline.create(dai.node.MonoCamera)
        right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
        stereo.initialConfig.setConfidenceThreshold(50)   # low = keep more pixels
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(True)                 # better close-range

        left.out.link(stereo.left)
        right.out.link(stereo.right)

        print("[DepthOverlay] Stereo nodes added to pipeline.")
        return stereo.depth

    except Exception as exc:
        print(f"[DepthOverlay] Stereo pipeline setup failed: {exc}")
        return None


def draw_overlay(frame: np.ndarray, depth_frame: np.ndarray) -> None:
    """Draw distance-coloured grid rectangles onto *frame* in-place.

    *depth_frame* is a uint16 numpy array from ``stereo.depth`` (values in mm).
    Cells within CRITICAL_MM draw red; cells within WARNING_MM draw orange.
    Cells with no valid depth or beyond WARNING_MM are skipped.
    """
    fh, fw = frame.shape[:2]
    dh, dw = depth_frame.shape

    for gy in range(GRID_Y):
        for gx in range(GRID_X):
            # Pixel region in the depth frame
            dr0 = int(gy * dh / GRID_Y)
            dr1 = int((gy + 1) * dh / GRID_Y)
            dc0 = int(gx * dw / GRID_X)
            dc1 = int((gx + 1) * dw / GRID_X)

            region = depth_frame[dr0:dr1, dc0:dc1]
            valid = region[region > 0]
            if len(valid) == 0:
                continue

            distance = int(np.percentile(valid, _DEPTH_PERCENTILE))
            if distance == 0 or distance > WARNING_MM:
                continue

            # Corresponding rectangle on the display frame (normalised grid)
            xmin = int(gx * fw / GRID_X)
            ymin = int(gy * fh / GRID_Y)
            xmax = int((gx + 1) * fw / GRID_X)
            ymax = int((gy + 1) * fh / GRID_Y)

            label = f"{distance / 1000:.2f}m"

            if distance < CRITICAL_MM:
                color = (0, 0, 255)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=3)
                cv2.putText(frame, label, (xmin + 4, ymin + 16),
                            _FONT, 0.38, color, 1, cv2.LINE_AA)
            else:
                color = (0, 140, 255)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=2)
                cv2.putText(frame, label, (xmin + 4, ymin + 16),
                            _FONT, 0.38, color, 1, cv2.LINE_AA)
