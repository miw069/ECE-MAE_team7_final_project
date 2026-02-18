"""Configuration constants for the person tracking system."""

# ── Detection ──────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"          # Ultralytics model name (downloaded automatically)
CONFIDENCE_THRESHOLD = 0.45        # Minimum detection confidence
PERSON_CLASS_ID = 0                # COCO class index for "person"

# ── Tracker ────────────────────────────────────────────────
TRACKER_TYPE = "bytetrack.yaml"    # Ultralytics tracker config

# ── State machine ──────────────────────────────────────────
LOST_TIMEOUT_FRAMES = 60           # Frames before giving up on a lost target (~2 sec at 30 fps)

# ── UI colours (BGR) ──────────────────────────────────────
COLOR_NORMAL = (0, 200, 0)         # Green  – unselected person
COLOR_LOCKED = (0, 100, 255)       # Orange – locked target
COLOR_LOST   = (0, 255, 255)       # Yellow – target lost, reacquiring
COLOR_TEXT_BG = (40, 40, 40)       # Dark grey panel
COLOR_WHITE  = (255, 255, 255)
COLOR_STATE_ARMED  = (200, 200, 0)
COLOR_STATE_LOCKED = (0, 180, 0)
COLOR_STATE_LOST   = (0, 180, 255)

# ── UI layout ─────────────────────────────────────────────
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS_NORMAL = 2
BOX_THICKNESS_TARGET = 3
INFO_PANEL_HEIGHT = 120            # Pixels at the bottom for status / input

# ── Web server ─────────────────────────────────────────────
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
JPEG_QUALITY = 80
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ── Re-ID ──────────────────────────────────────────────────
REID_MODEL = "mobilenetv3_small_100"
REID_SIMILARITY_THRESHOLD = 0.65
REID_CHECK_INTERVAL = 5            # Run Re-ID every N lost frames
