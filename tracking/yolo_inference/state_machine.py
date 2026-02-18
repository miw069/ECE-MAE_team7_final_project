"""Tracking state machine: manages target selection and lock lifecycle."""

from enum import Enum
from typing import List, Optional

from detector import TrackedPerson
import config


class TrackingState(Enum):
    ARMED = "ARMED"    # Waiting for the user to select a target
    LOCKED = "LOCKED"  # Actively tracking the selected target
    LOST = "LOST"      # Target temporarily lost; attempting to reacquire


class TargetStateMachine:
    """Simple state machine that keeps track of the selected person.

    State transitions
    -----------------
    ARMED  --(user selects ID)--> LOCKED
    LOCKED --(target missing)---> LOST
    LOST   --(target reappears)-> LOCKED
    LOST   --(Re-ID match)------> LOCKED  (new track_id)
    LOST   --(timeout)----------> ARMED
    LOCKED --(user cancels)-----> ARMED
    """

    def __init__(
        self,
        lost_timeout: int = config.LOST_TIMEOUT_FRAMES,
        reid_manager=None,
    ):
        self.state: TrackingState = TrackingState.ARMED
        self.target_id: Optional[int] = None
        self.lost_frames: int = 0
        self.lost_timeout: int = lost_timeout
        self.reid_manager = reid_manager

        # The current bbox of the target (or last known bbox if LOST)
        self.target_bbox: Optional[tuple] = None

        # Appearance embedding of the locked target (used for Re-ID)
        self.target_embedding: Optional[object] = None

    # ── public API ──────────────────────────────────────────

    def select_target(self, track_id: int, persons: List[TrackedPerson]) -> bool:
        """Attempt to lock onto a person with the given track_id.

        Returns True if the ID was found among current tracks and locked.
        """
        for person in persons:
            if person.track_id == track_id:
                self.target_id = track_id
                self.target_bbox = person.bbox
                self.state = TrackingState.LOCKED
                self.lost_frames = 0
                # Reset embedding so it will be re-extracted on next LOCKED frame
                self.target_embedding = None
                print(f"[State] LOCKED onto target ID {track_id}")
                return True

        print(f"[State] ID {track_id} not found in current tracks.")
        return False

    def cancel(self) -> None:
        """Cancel current tracking and return to ARMED."""
        prev = self.state
        self.state = TrackingState.ARMED
        self.target_id = None
        self.target_bbox = None
        self.lost_frames = 0
        self.target_embedding = None
        print(f"[State] {prev.value} -> ARMED (cancelled)")

    def update(
        self,
        persons: List[TrackedPerson],
        frame=None,
    ) -> Optional[TrackedPerson]:
        """Called every frame with the current tracked persons.

        Args:
            persons: Detected/tracked persons this frame.
            frame:   Raw BGR frame (numpy array). Required for Re-ID embedding
                     extraction; if None, Re-ID is skipped.

        Returns:
            The TrackedPerson for the target if LOCKED, else None.
        """
        if self.state == TrackingState.ARMED:
            return None

        # Look for our target in the current tracks
        target = next(
            (p for p in persons if p.track_id == self.target_id), None
        )

        if self.state == TrackingState.LOCKED:
            if target is not None:
                self.target_bbox = target.bbox
                self.lost_frames = 0

                # Extract appearance embedding once per lock acquisition
                if (
                    self.target_embedding is None
                    and self.reid_manager is not None
                    and frame is not None
                ):
                    emb = self.reid_manager.extract(frame, target.bbox)
                    if emb is not None:
                        self.target_embedding = emb
                        print("[State] Target embedding extracted.")

                return target
            else:
                # Just lost the target
                self.state = TrackingState.LOST
                self.lost_frames = 1
                print(f"[State] LOCKED -> LOST (target {self.target_id} disappeared)")
                return None

        if self.state == TrackingState.LOST:
            if target is not None:
                # ByteTrack reacquired with the same ID
                self.state = TrackingState.LOCKED
                self.target_bbox = target.bbox
                self.lost_frames = 0
                print(f"[State] LOST -> LOCKED (target {self.target_id} reacquired by tracker)")
                return target
            else:
                self.lost_frames += 1

                # Attempt Re-ID every REID_CHECK_INTERVAL frames
                if (
                    self.reid_manager is not None
                    and self.target_embedding is not None
                    and frame is not None
                    and self.lost_frames % config.REID_CHECK_INTERVAL == 0
                    and persons
                ):
                    candidates = [
                        (p.track_id, self.reid_manager.extract(frame, p.bbox))
                        for p in persons
                    ]
                    matched_id = self.reid_manager.find_best_match(
                        self.target_embedding, candidates
                    )
                    if matched_id is not None:
                        print(
                            f"[State] LOST -> LOCKED via Re-ID "
                            f"(new track_id={matched_id}, was {self.target_id})"
                        )
                        self.select_target(matched_id, persons)
                        return next(
                            (p for p in persons if p.track_id == matched_id), None
                        )

                if self.lost_frames >= self.lost_timeout:
                    print(
                        f"[State] LOST -> ARMED (target {self.target_id} lost for "
                        f"{self.lost_frames} frames, giving up)"
                    )
                    self.target_id = None
                    self.target_bbox = None
                    self.target_embedding = None
                    self.state = TrackingState.ARMED

                return None

        return None
