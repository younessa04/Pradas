"""
PRADAS Centralized Configuration.

All configurable parameters for the system. Uses environment variables
with sensible defaults. No hardcoded machine-specific paths.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


def _get_env(key: str, default: str = "") -> str:
    """Get environment variable with PRADAS_ prefix."""
    return os.environ.get(f"PRADAS_{key}", default)


def _get_env_float(key: str, default: float) -> float:
    """Get float environment variable."""
    val = _get_env(key, "")
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _get_env_int(key: str, default: int) -> int:
    """Get integer environment variable."""
    val = _get_env(key, "")
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable."""
    val = _get_env(key, "").lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return default


@dataclass
class DetectionConfig:
    """Object detection and pose estimation settings."""
    yolo_pose_model: str = field(default_factory=lambda: _get_env("YOLO_POSE_MODEL", "yolov8n-pose.pt"))
    confidence_threshold: float = field(default_factory=lambda: _get_env_float("CONFIDENCE_THRESHOLD", 0.30))
    iou_threshold: float = field(default_factory=lambda: _get_env_float("IOU_THRESHOLD", 0.45))
    max_detections: int = field(default_factory=lambda: _get_env_int("MAX_DETECTIONS", 50))
    imgsz: int = field(default_factory=lambda: _get_env_int("IMGSZ", 960))
    person_class_id: int = 0  # COCO person class


@dataclass
class TrackingConfig:
    """Multi-object tracking settings."""
    tracker_type: str = "bytetrack"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30  # Frames to keep lost tracks
    match_thresh: float = 0.8
    max_history_length: int = 90  # Max position history per track


@dataclass
class DepthConfig:
    """Monocular depth estimation settings."""
    midas_model_type: str = field(default_factory=lambda: _get_env("MIDAS_MODEL_TYPE", "MiDaS_small"))
    depth_input_size: int = field(default_factory=lambda: _get_env_int("DEPTH_INPUT_SIZE", 256))
    use_median_depth: bool = True
    roi_shrink_factor: float = 0.2  # Shrink bbox to avoid background


@dataclass
class DangerZoneConfig:
    """Danger zone polygon configuration.

    Coordinates are normalized [0, 1] relative to frame dimensions.
    Default polygon covers a trapezoidal region ahead of the vehicle.
    """
    enabled: bool = True
    # Default: trapezoidal zone in front of vehicle (normalized coords)
    polygon_normalized: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.2, 0.5),   # top-left
        (0.8, 0.5),   # top-right
        (0.95, 1.0),  # bottom-right
        (0.05, 1.0),  # bottom-left
    ])
    approach_margin_px: int = 50  # Pixels outside zone considered "approaching"


@dataclass
class RiskConfig:
    """Risk assessment and TTC settings."""
    # Risk score thresholds (0-1 scale)
    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.4

    # TTC thresholds (seconds)
    ttc_critical: float = 2.0
    ttc_warning: float = 4.0
    ttc_min_observations: int = 5  # Minimum frames before TTC is computed

    # Temporal smoothing
    risk_smoothing_window: int = 5  # Frames for risk level smoothing
    alert_cooldown_seconds: float = 2.0  # Min time between alerts for same track
    hysteresis_frames: int = 3  # Frames before risk level can change

    # Scoring weights
    weight_danger_zone: float = 0.30
    weight_depth_trend: float = 0.20
    weight_ttc: float = 0.25
    weight_intent: float = 0.15
    weight_confidence: float = 0.10


@dataclass
class NightEnhancementConfig:
    """Low-light enhancement settings."""
    mode: str = field(default_factory=lambda: _get_env("NIGHT_MODE", "auto"))  # auto, always, never
    luminance_threshold: float = field(default_factory=lambda: _get_env_float("LUMINANCE_THRESHOLD", 80.0))
    gamma: float = field(default_factory=lambda: _get_env_float("GAMMA", 0.6))
    clahe_clip_limit: float = field(default_factory=lambda: _get_env_float("CLAHE_CLIP", 3.0))
    clahe_grid_size: Tuple[int, int] = (8, 8)


@dataclass
class VideoConfig:
    """Video I/O settings."""
    source: str = field(default_factory=lambda: _get_env("VIDEO_SOURCE", "0"))
    output_path: str = field(default_factory=lambda: _get_env("OUTPUT_VIDEO", ""))
    display: bool = field(default_factory=lambda: not _get_env_bool("HEADLESS", False))
    frame_skip: int = field(default_factory=lambda: _get_env_int("FRAME_SKIP", 1))
    output_fps: float = 25.0
    output_codec: str = "mp4v"


@dataclass
class APIConfig:
    """API server settings."""
    host: str = field(default_factory=lambda: _get_env("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _get_env_int("API_PORT", 8000))
    ws_enabled: bool = field(default_factory=lambda: _get_env_bool("WS_ENABLED", False))
    ws_max_queue_size: int = 100
    ws_update_interval: float = 0.1  # Seconds between WS broadcasts


@dataclass
class Settings:
    """Master configuration container for the PRADAS system."""
    device: str = field(default_factory=lambda: _get_env("DEVICE", "auto"))
    log_level: str = field(default_factory=lambda: _get_env("LOG_LEVEL", "INFO"))

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    danger_zone: DangerZoneConfig = field(default_factory=DangerZoneConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    night_enhancement: NightEnhancementConfig = field(default_factory=NightEnhancementConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    api: APIConfig = field(default_factory=APIConfig)

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def validate(self) -> List[str]:
        """Validate configuration, return list of warnings."""
        warnings = []
        if self.detection.confidence_threshold < 0.1:
            warnings.append("Very low detection confidence threshold may cause excessive false positives.")
        if self.detection.confidence_threshold > 0.95:
            warnings.append("Very high detection confidence threshold may miss valid detections.")
        if self.video.frame_skip < 1:
            warnings.append("Frame skip must be >= 1, setting to 1.")
            self.video.frame_skip = 1
        if len(self.danger_zone.polygon_normalized) < 3:
            warnings.append("Danger zone polygon needs at least 3 vertices.")
        return warnings


def get_settings() -> Settings:
    """Create a Settings instance from environment/defaults."""
    return Settings()
