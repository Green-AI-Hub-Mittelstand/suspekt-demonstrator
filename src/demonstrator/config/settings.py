from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

CONFIG_DIR: Path = PROJECT_ROOT / "config"
ROI_CONFIG_PATH: Path = CONFIG_DIR / "center_roi.json"
MODELS_DIR: Path = PROJECT_ROOT / "models"


# ============================================================
# CAMERA / YOLO SETTINGS
# ============================================================
OAK_FRAME_WIDTH: int = 640
OAK_FRAME_HEIGHT: int = 480
OAK_FPS: int = 20
OAK_ANTI_BANDING: str = "50hz"
OAK_MANUAL_EXPOSURE_US: Optional[int] = None
OAK_MANUAL_ISO: Optional[int] = None
OAK_LUMA_DENOISE: Optional[int] = 2
OAK_CHROMA_DENOISE: Optional[int] = 2
OAK_SHARPNESS: Optional[int] = 1

# Convert the Path objects to strings to match your existing type hints and downstream code
YOLO_ENGINE_LEFT: str = str(MODELS_DIR / "best_nubsDetection_42kl_320_FP16_detect.engine")
YOLO_ENGINE_RIGHT: str = str(MODELS_DIR / "best_nubsDetection_42kl_320_FP16_detect.engine")

USB_DEVICE_INDEX: int = 0
USB_CAPTURE_WIDTH: int = 640
USB_CAPTURE_HEIGHT: int = 480
USB_CAPTURE_FPS: int = 20

YOLO_ENGINE_CENTER: str = str(MODELS_DIR / "best_demonstrator_42_kl_320_FP16_detect.engine")

YOLO_MODEL_INPUT_SIZE: int = 320
YOLO_CONF_THRESH: float = 0.4
YOLO_IOU_THRESH: float = 0.45
YOLO_MAX_DET: int = 300
YOLO_DEVICE: str = "cuda"

MJPEG_BOUNDARY: bytes = b"--frameboundary"
SKIP_FRAMES: int = 8
BATCH_SIZE: int = 1
ARUCO_ENABLED: bool = True
ARUCO_SKIP_FRAMES: int = 120
# Keep detections visible between inference passes (seconds).
DETECTION_PERSIST_SECONDS: float = 0.75
# Optional hard cap for per-camera inference rate. Set 0 to disable.
INFERENCE_MAX_FPS: float = 5.0


# ============================================================
# GAME / PARTS CONFIGURATION
# ============================================================
DEFAULT_CENTER_CAMERA_ROI_REL: Optional[Tuple[float, float, float, float]] = (
    0.05,
    0.02,
    0.9,
    0.9,
)

ALL_PART_DEFINITIONS: List[Dict[str, object]] = [
    {
        "id": "gerade",
        "display_name": "Gerade",
        "aliases": {"gerade", "straight", "straight_bar", "geraden"},
    },
    {
        "id": "versetzte_gerade",
        "display_name": "Versetzte Gerade",
        "aliases": {"versetzte gerade", "versetzte_gerade", "offset", "versetzt"},
    },
    {
        "id": "diagonale",
        "display_name": "Diagonale",
        "aliases": {"diagonale", "diagonal", "diagonal_bar"},
    },
    {
        "id": "mutternstab",
        "display_name": "Mutternstab",
        "aliases": {"mutternstab", "threaded_rod", "mutterstab"},
    },
    {
        "id": "noppenscheiben",
        "display_name": "Noppenscheiben",
        "aliases": {"noppenscheiben", "noppen", "stud", "noppen_stein"},
    },
    {
        "id": "schraube",
        "display_name": "Schraube",
        "aliases": {"schraube", "screw"},
    },
]

GAME_PART_IDS: List[str] = [part["id"] for part in ALL_PART_DEFINITIONS]
_env_part_ids = os.getenv("GAME_PART_IDS")
if _env_part_ids:
    GAME_PART_IDS = [
        cls.strip()
        for cls in _env_part_ids.split(",")
        if cls.strip()
    ]

_known_part_ids = {part["id"] for part in ALL_PART_DEFINITIONS}
_unknown_part_ids = [cls for cls in GAME_PART_IDS if cls not in _known_part_ids]
if _unknown_part_ids:
    print(
        "Warnung: Unbekannte Objektklassen in GAME_PART_IDS ignoriert: "
        + ", ".join(sorted(set(_unknown_part_ids)))
    )

GAME_PART_IDS = list(
    dict.fromkeys(cls for cls in GAME_PART_IDS if cls in _known_part_ids)
)
if not GAME_PART_IDS:
    GAME_PART_IDS = [part["id"] for part in ALL_PART_DEFINITIONS]

PART_CATALOG: List[Dict[str, object]] = [
    part for part in ALL_PART_DEFINITIONS if part["id"] in GAME_PART_IDS
]
PART_LOOKUP: Dict[str, Dict[str, object]] = {part["id"]: part for part in PART_CATALOG}

INVENTORY_GERADE_BINS_MM: List[float] = [250.0, 300.0, 360.0, 460.0]
INVENTORY_GERADE_BIN_TOLERANCE_MM: float = 15.0
INVENTORY_SORT_DURATION_S: float = 90.0


# ============================================================
# AUTH / SESSION
# ============================================================
ADMIN_USERNAME: str = "admin"
ADMIN_PASSWORD: str = "gaih"
SESSION_COOKIE_NAME: str = "admin_session"
SESSION_TTL_SECONDS: int = 8 * 3600


# ============================================================
# HELPERS
# ============================================================
def normalise_roi_tuple(value: Optional[Iterable[float]]) -> Optional[Tuple[float, float, float, float]]:
    """Utility that clamps and validates ROI tuples to [0, 1] boundaries."""
    if value is None:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)
