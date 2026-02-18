#!/usr/bin/env python3
"""Person Tracking System – main entry point.

Usage:
    python main.py                    # OAK-D Lite camera (default)
    python main.py --model yolov8s.pt # use a different YOLO model
"""

import argparse
import sys

import cv2

from camera import OakDLiteCamera
from detector import PersonDetector
from state_machine import TargetStateMachine
from ui import TrackingUI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Person tracking with YOLO + ByteTrack")
    p.add_argument(
        "--model",
        default=None,
        help="Ultralytics YOLO model name/path (e.g. yolov8n.pt, yolov8s.pt). "
             "Default: from config.py",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold. Default: from config.py",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Inference device: 'cuda', 'cpu', or 'cuda:0'. Default: auto-detect.",
    )
    p.add_argument(
        "--lost-timeout",
        type=int,
        default=None,
        help="Frames to wait before giving up on a lost target. Default: from config.py",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Initialise components ──────────────────────────────
    detector_kwargs = {}
    if args.model:
        detector_kwargs["model_name"] = args.model
    if args.conf is not None:
        detector_kwargs["conf_threshold"] = args.conf
    if args.device:
        detector_kwargs["device"] = args.device

    detector = PersonDetector(**detector_kwargs)

    sm_kwargs = {}
    if args.lost_timeout is not None:
        sm_kwargs["lost_timeout"] = args.lost_timeout
    sm = TargetStateMachine(**sm_kwargs)

    ui = TrackingUI()

    cam = OakDLiteCamera()

    print("\n──────────────────────────────────────────────")
    print("  Person Tracker running.  Controls:")
    print("    [0-9]       Type a track ID")
    print("    [Enter]     Lock onto that ID")
    print("    [C]         Cancel / deselect")
    print("    [Backspace] Delete last digit")
    print("    [Q / Esc]   Quit")
    print("──────────────────────────────────────────────\n")

    # ── Main loop ──────────────────────────────────────────
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            # 1. Detect + track persons
            persons = detector.update(frame)

            # 2. Update state machine
            sm.update(persons)

            # 3. Render UI
            display = ui.render(frame, persons, sm)

            cv2.imshow(ui.WINDOW_NAME, display)

            # 4. Handle keyboard
            key = cv2.waitKey(1)
            action = ui.process_key(key)

            if action == "quit":
                break
            elif action == "cancel":
                sm.cancel()
                ui.flash("Tracking cancelled.")
            elif action and action.startswith("select:"):
                try:
                    tid = int(action.split(":")[1])
                except ValueError:
                    ui.flash("Invalid ID.")
                    continue
                if sm.select_target(tid, persons):
                    ui.flash(f"Locked onto ID {tid}")
                else:
                    ui.flash(f"ID {tid} not visible – try again.")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("[Main] Shutdown complete.")


if __name__ == "__main__":
    main()
