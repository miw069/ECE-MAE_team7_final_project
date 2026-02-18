"""Person detection and multi-object tracking using YOLO + ByteTrack."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

import config


@dataclass
class TrackedPerson:
    """A single tracked person in a frame."""

    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class PersonDetector:
    """Wraps Ultralytics YOLO for person detection + ByteTrack tracking.

    Uses model.track() which performs detection and multi-object tracking
    in a single call, maintaining track IDs across frames.
    """

    def __init__(
        self,
        model_name: str = config.YOLO_MODEL,
        conf_threshold: float = config.CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.conf_threshold = conf_threshold

        print(f"[Detector] Loading {model_name} on {device} ...")
        self.model = YOLO(model_name)
        print(f"[Detector] Model loaded.")

    def update(self, frame: np.ndarray) -> List[TrackedPerson]:
        """Run detection + tracking on a single BGR frame.

        Args:
            frame: BGR image (H, W, 3) as numpy array.

        Returns:
            List of TrackedPerson objects for every person detected in this frame.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            classes=[config.PERSON_CLASS_ID],
            verbose=False,
            tracker=config.TRACKER_TYPE,
            device=self.device,
        )

        persons: List[TrackedPerson] = []

        if not results or results[0].boxes is None:
            return persons

        boxes = results[0].boxes

        for box in boxes:
            # box.id is None when the tracker hasn't assigned an ID yet
            if box.id is None:
                continue

            track_id = int(box.id.item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf.item())

            persons.append(
                TrackedPerson(
                    track_id=track_id,
                    bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    confidence=conf,
                )
            )

        return persons
