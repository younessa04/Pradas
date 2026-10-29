"""
PRADAS Pedestrian Intent Analyzer.

Estimates pedestrian movement state and crossing intent from temporal
observations. Uses image-plane motion analysis — NOT real-world trajectory
estimation (unless camera calibration is provided).

Important: Observable motion is not equivalent to inferred intent.
This module provides evidence-based state estimates, not certainties.
"""

import math
import numpy as np
from enum import Enum
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.settings import DangerZoneConfig
from src.tracker import TrackedPedestrian
from utils.geometry import (
    point_in_polygon, denormalize_polygon,
    compute_velocity, distance_to_polygon_edge
)
from utils.logging_config import get_logger

logger = get_logger("intent_analyzer")


class IntentState(str, Enum):
    """Pedestrian movement/intent classification."""
    UNKNOWN = "UNKNOWN"
    STATIONARY = "STATIONARY"
    MOVING = "MOVING"
    POSSIBLE_CROSSING = "POSSIBLE_CROSSING"
    ENTERING_DANGER_ZONE = "ENTERING_DANGER_ZONE"


@dataclass
class IntentResult:
    """
    Result of intent analysis for a single pedestrian.

    All direction/velocity estimates are in IMAGE-PLANE coordinates
    unless camera calibration is available.
    """
    track_id: int
    state: IntentState = IntentState.UNKNOWN
    confidence: float = 0.0

    # Image-plane motion (pixels/frame or pixels/second if timestamps available)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    speed: float = 0.0
    direction_degrees: float = 0.0  # 0=right, 90=down, etc.

    # Danger zone status
    in_danger_zone: bool = False
    approaching_danger_zone: bool = False
    distance_to_zone_px: float = float('inf')

    # Bounding box dynamics
    bbox_height_change_rate: float = 0.0  # + means growing (approaching)
    bbox_area_change_rate: float = 0.0

    # Pose-based indicators
    torso_orientation: Optional[str] = None  # facing_left, facing_right, facing_camera

    # Smoothed state history
    _state_history: deque = field(default_factory=lambda: deque(maxlen=10))

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "track_id": self.track_id,
            "state": self.state.value,
            "confidence": round(self.confidence, 3),
            "velocity_x": round(self.velocity_x, 2),
            "velocity_y": round(self.velocity_y, 2),
            "speed": round(self.speed, 2),
            "direction_degrees": round(self.direction_degrees, 1),
            "in_danger_zone": self.in_danger_zone,
            "approaching_danger_zone": self.approaching_danger_zone,
            "distance_to_zone_px": round(self.distance_to_zone_px, 1)
                if self.distance_to_zone_px != float('inf') else None,
            "torso_orientation": self.torso_orientation,
        }


class PedestrianIntentAnalyzer:
    """
    Analyzes pedestrian movement patterns and estimates crossing intent.

    Uses temporal observations (position history, bbox dynamics, pose)
    rather than single-frame analysis.
    """

    def __init__(
        self,
        danger_zone_config: Optional[DangerZoneConfig] = None,
        frame_width: int = 1920,
        frame_height: int = 1080
    ):
        """
        Args:
            danger_zone_config: Danger zone configuration.
            frame_width: Video frame width for coordinate mapping.
            frame_height: Video frame height for coordinate mapping.
        """
        self.dz_config = danger_zone_config or DangerZoneConfig()
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Denormalize danger zone polygon to pixel coordinates
        self._danger_zone_px: List[Tuple[int, int]] = []
        self._update_danger_zone()

        # Per-track intent results (smoothed)
        self._intent_cache: Dict[int, IntentResult] = {}

        # Thresholds for motion classification (image-plane)
        self._stationary_threshold = 2.0   # pixels/frame
        self._moving_threshold = 5.0       # pixels/frame
        self._approach_margin = self.dz_config.approach_margin_px

        logger.info(f"IntentAnalyzer initialized: "
                    f"frame={frame_width}x{frame_height}, "
                    f"danger_zone_vertices={len(self.dz_config.polygon_normalized)}")

    def _update_danger_zone(self) -> None:
        """Recompute danger zone polygon in pixel coordinates."""
        if self.dz_config.enabled and len(self.dz_config.polygon_normalized) >= 3:
            self._danger_zone_px = denormalize_polygon(
                self.dz_config.polygon_normalized,
                self.frame_width,
                self.frame_height
            )

    def update_frame_size(self, width: int, height: int) -> None:
        """Update frame dimensions and recompute danger zone."""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self._update_danger_zone()

    @property
    def danger_zone_pixels(self) -> List[Tuple[int, int]]:
        """Danger zone polygon in pixel coordinates."""
        return self._danger_zone_px

    def analyze(
        self,
        tracked_pedestrians: List[TrackedPedestrian]
    ) -> Dict[int, IntentResult]:
        """
        Analyze intent for all tracked pedestrians.

        Args:
            tracked_pedestrians: List of tracked pedestrians with history.

        Returns:
            Dict mapping track_id to IntentResult.
        """
        results: Dict[int, IntentResult] = {}

        for ped in tracked_pedestrians:
            result = self._analyze_single(ped)
            results[ped.track_id] = result

        # Clean up cache for tracks that no longer exist
        active_ids = {p.track_id for p in tracked_pedestrians}
        expired = [tid for tid in self._intent_cache if tid not in active_ids]
        for tid in expired:
            del self._intent_cache[tid]

        self._intent_cache.update(results)
        return results

    def _analyze_single(self, ped: TrackedPedestrian) -> IntentResult:
        """Analyze a single tracked pedestrian."""
        result = IntentResult(track_id=ped.track_id)

        # 1. Compute velocity from position history
        positions = list(ped.history.centers)
        timestamps = list(ped.history.timestamps) if ped.history.timestamps else None
        vx, vy = compute_velocity(positions, timestamps, window=5)
        result.velocity_x = vx
        result.velocity_y = vy
        result.speed = math.sqrt(vx * vx + vy * vy)
        result.direction_degrees = math.degrees(math.atan2(vy, vx)) % 360

        # 2. Bounding box dynamics
        if len(ped.history.bbox_heights) >= 3:
            heights = list(ped.history.bbox_heights)
            result.bbox_height_change_rate = (heights[-1] - heights[-3]) / 3.0

        if len(ped.history.bbox_areas) >= 3:
            areas = list(ped.history.bbox_areas)
            result.bbox_area_change_rate = (areas[-1] - areas[-3]) / 3.0

        # 3. Danger zone analysis
        if self.dz_config.enabled and len(self._danger_zone_px) >= 3:
            center = ped.center
            center_float = (float(center[0]), float(center[1]))

            # Check if inside the danger zone
            result.in_danger_zone = point_in_polygon(
                center_float,
                [(float(x), float(y)) for x, y in self._danger_zone_px]
            )

            # Distance to zone boundary
            result.distance_to_zone_px = distance_to_polygon_edge(
                center_float,
                [(float(x), float(y)) for x, y in self._danger_zone_px]
            )

            # Check if approaching
            if not result.in_danger_zone:
                result.approaching_danger_zone = (
                    result.distance_to_zone_px < self._approach_margin
                )

        # 4. Torso orientation from keypoints
        from src.detector_pose import PedestrianDetection
        if ped.keypoints is not None:
            temp_det = PedestrianDetection(
                bbox=ped.bbox, confidence=ped.confidence,
                keypoints=ped.keypoints
            )
            result.torso_orientation = temp_det.estimate_torso_orientation()

        # 5. Classify movement state
        result.state, result.confidence = self._classify_state(result, ped)

        # 6. Apply temporal smoothing
        result = self._smooth_state(result)

        return result

    def _classify_state(
        self,
        result: IntentResult,
        ped: TrackedPedestrian
    ) -> Tuple[IntentState, float]:
        """
        Classify pedestrian movement state from combined evidence.

        Returns:
            (IntentState, confidence) tuple.
        """
        # Need minimum history for reliable classification
        if len(ped.history.centers) < 3:
            return (IntentState.UNKNOWN, 0.3)

        speed = result.speed

        # Stationary check
        if speed < self._stationary_threshold:
            return (IntentState.STATIONARY, 0.8)

        # Inside danger zone and moving
        if result.in_danger_zone:
            return (IntentState.POSSIBLE_CROSSING, 0.75)

        # Approaching danger zone with significant movement
        if result.approaching_danger_zone and speed > self._moving_threshold:
            return (IntentState.ENTERING_DANGER_ZONE, 0.65)

        # General movement
        if speed > self._moving_threshold:
            # Check if moving TOWARD the danger zone
            if self._is_moving_toward_zone(ped, result):
                return (IntentState.ENTERING_DANGER_ZONE, 0.6)
            return (IntentState.MOVING, 0.7)

        return (IntentState.MOVING, 0.5)

    def _is_moving_toward_zone(
        self,
        ped: TrackedPedestrian,
        result: IntentResult
    ) -> bool:
        """
        Check if pedestrian velocity vector points toward the danger zone.

        This is an image-plane estimate only.
        """
        if not self._danger_zone_px or len(self._danger_zone_px) < 3:
            return False

        # Zone centroid
        zone_cx = sum(x for x, y in self._danger_zone_px) / len(self._danger_zone_px)
        zone_cy = sum(y for x, y in self._danger_zone_px) / len(self._danger_zone_px)

        cx, cy = ped.center
        dx_to_zone = zone_cx - cx
        dy_to_zone = zone_cy - cy

        # Dot product of velocity and direction to zone
        dot = result.velocity_x * dx_to_zone + result.velocity_y * dy_to_zone
        return dot > 0

    def _smooth_state(self, result: IntentResult) -> IntentResult:
        """Apply temporal smoothing to prevent flickering."""
        if result.track_id in self._intent_cache:
            prev = self._intent_cache[result.track_id]
            prev._state_history.append(result.state)

            # Require multiple frames to transition to higher-risk states
            if len(prev._state_history) >= 3:
                recent = list(prev._state_history)[-3:]
                # If majority agrees, keep the new state; otherwise, keep previous
                if recent.count(result.state) < 2:
                    result.state = prev.state
                    result.confidence *= 0.9

            result._state_history = prev._state_history
        else:
            result._state_history = deque(maxlen=10)
            result._state_history.append(result.state)

        return result

    @property
    def danger_zone_pixels(self) -> List[Tuple[int, int]]:
        """Get danger zone polygon in pixel coordinates."""
        return self._danger_zone_px
