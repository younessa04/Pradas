"""
PRADAS Risk Assessment and Time-to-Collision Engine.

Core decision-making module that combines evidence from detection, tracking,
depth estimation, and intent analysis to produce risk assessments.

IMPORTANT CONSTRAINTS:
    - Relative depth is NOT metric distance.
    - Bounding-box expansion is NOT calibrated physical TTC.
    - Missing data must NOT be interpreted as LOW risk.
    - This is NOT a certified ADAS — it CANNOT control vehicle systems.
"""

import time
import math
import numpy as np
from enum import Enum
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.settings import RiskConfig, DangerZoneConfig
from src.tracker import TrackedPedestrian
from src.intent_analyzer import IntentState, IntentResult
from utils.logging_config import get_logger

logger = get_logger("risk_engine")


class RiskLevel(str, Enum):
    """Pedestrian risk classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DangerZoneStatus(str, Enum):
    """Pedestrian position relative to danger zone."""
    OUTSIDE = "OUTSIDE"
    APPROACHING = "APPROACHING"
    INSIDE = "INSIDE"
    LEAVING = "LEAVING"
    UNKNOWN = "UNKNOWN"


class TTCMethod(str, Enum):
    """Method used to estimate TTC."""
    BBOX_EXPANSION = "bbox_expansion"
    DEPTH_TREND = "depth_trend"
    CALIBRATED_METRIC = "calibrated_metric"
    UNAVAILABLE = "unavailable"


@dataclass
class RiskAssessment:
    """
    Complete risk assessment for a single pedestrian.

    All fields are JSON-serializable via to_dict().
    """
    track_id: int
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0

    # TTC
    ttc_seconds: Optional[float] = None
    ttc_method: TTCMethod = TTCMethod.UNAVAILABLE
    ttc_confidence: float = 0.0

    # Depth
    relative_depth: Optional[float] = None
    distance_meters: Optional[float] = None  # Only if calibrated

    # Intent
    intent_state: IntentState = IntentState.UNKNOWN
    danger_zone_status: DangerZoneStatus = DangerZoneStatus.UNKNOWN

    # Explanations
    reason_codes: List[str] = field(default_factory=list)

    # Metadata
    timestamp: str = ""
    frame_index: int = 0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "track_id": self.track_id,
            "risk_level": self.risk_level.value,
            "risk_score": round(self.risk_score, 3),
            "ttc_seconds": round(self.ttc_seconds, 2) if self.ttc_seconds is not None else None,
            "ttc_method": self.ttc_method.value,
            "ttc_confidence": round(self.ttc_confidence, 3),
            "relative_depth": round(self.relative_depth, 3) if self.relative_depth is not None else None,
            "distance_meters": round(self.distance_meters, 2) if self.distance_meters is not None else None,
            "intent_state": self.intent_state.value,
            "danger_zone_status": self.danger_zone_status.value,
            "reason_codes": self.reason_codes,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
        }


class PedestrianRiskEngine:
    """
    Risk assessment engine combining multiple evidence sources.

    Uses a configurable weighted scoring model with temporal smoothing,
    hysteresis, and alert cooldowns to produce stable risk assessments.
    """

    def __init__(
        self,
        risk_config: Optional[RiskConfig] = None,
        danger_zone_config: Optional[DangerZoneConfig] = None
    ):
        """
        Args:
            risk_config: Risk thresholds and weights.
            danger_zone_config: Danger zone configuration.
        """
        self.config = risk_config or RiskConfig()
        self.dz_config = danger_zone_config or DangerZoneConfig()

        # Per-track risk history for smoothing and hysteresis
        self._risk_history: Dict[int, deque] = {}
        self._last_alert_time: Dict[int, float] = {}
        self._previous_risk: Dict[int, RiskLevel] = {}

        logger.info(f"RiskEngine initialized: "
                    f"high_threshold={self.config.high_risk_threshold}, "
                    f"medium_threshold={self.config.medium_risk_threshold}")

    def assess(
        self,
        tracked_pedestrians: List[TrackedPedestrian],
        intent_results: Dict[int, IntentResult],
        depth_values: Dict[int, Optional[float]],
        metric_distances: Dict[int, Optional[float]],
        frame_index: int,
        timestamp: Optional[str] = None
    ) -> List[RiskAssessment]:
        """
        Assess risk for all tracked pedestrians.

        Args:
            tracked_pedestrians: Currently tracked pedestrians.
            intent_results: Intent analysis results per track.
            depth_values: Relative depth per track.
            metric_distances: Calibrated metric distances per track (may be None).
            frame_index: Current frame index.
            timestamp: ISO-format timestamp string.

        Returns:
            List of RiskAssessment objects.
        """
        ts = timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        assessments: List[RiskAssessment] = []

        for ped in tracked_pedestrians:
            tid = ped.track_id
            intent = intent_results.get(tid, IntentResult(track_id=tid))
            rel_depth = depth_values.get(tid)
            metric_dist = metric_distances.get(tid)

            assessment = self._assess_single(
                ped, intent, rel_depth, metric_dist, frame_index, ts
            )
            assessments.append(assessment)

        # Clean up history for expired tracks
        active_ids = {p.track_id for p in tracked_pedestrians}
        expired = [tid for tid in self._risk_history if tid not in active_ids]
        for tid in expired:
            del self._risk_history[tid]
            self._previous_risk.pop(tid, None)
            self._last_alert_time.pop(tid, None)

        return assessments

    def _assess_single(
        self,
        ped: TrackedPedestrian,
        intent: IntentResult,
        relative_depth: Optional[float],
        metric_distance: Optional[float],
        frame_index: int,
        timestamp: str
    ) -> RiskAssessment:
        """Assess risk for a single pedestrian."""
        tid = ped.track_id
        reasons: List[str] = []

        # Initialize risk assessment
        assessment = RiskAssessment(
            track_id=tid,
            relative_depth=relative_depth,
            distance_meters=metric_distance,
            intent_state=intent.state,
            timestamp=timestamp,
            frame_index=frame_index,
        )

        # --- Danger zone status ---
        dz_score = 0.0
        if intent.in_danger_zone:
            assessment.danger_zone_status = DangerZoneStatus.INSIDE
            dz_score = 1.0
            reasons.append("INSIDE_DANGER_ZONE")
        elif intent.approaching_danger_zone:
            assessment.danger_zone_status = DangerZoneStatus.APPROACHING
            dz_score = 0.6
            reasons.append("APPROACHING_DANGER_ZONE")
        else:
            assessment.danger_zone_status = DangerZoneStatus.OUTSIDE
            dz_score = 0.0

        # --- TTC estimation ---
        ttc_score = 0.0
        ttc_val, ttc_method, ttc_conf = self._estimate_ttc(
            ped, relative_depth, metric_distance
        )
        assessment.ttc_seconds = ttc_val
        assessment.ttc_method = ttc_method
        assessment.ttc_confidence = ttc_conf

        if ttc_val is not None and ttc_conf > 0.3:
            if ttc_val < self.config.ttc_critical:
                ttc_score = 1.0
                reasons.append("CRITICAL_TTC")
            elif ttc_val < self.config.ttc_warning:
                ttc_score = 0.6
                reasons.append("SHORT_TTC")
            else:
                ttc_score = 0.2

        # --- Depth trend score ---
        depth_score = 0.0
        if relative_depth is not None and len(ped.history.depths) >= 3:
            depths = list(ped.history.depths)
            # Positive trend = getting closer (higher relative depth)
            trend = (depths[-1] - depths[-3]) / 3.0
            if trend > 0.02:
                depth_score = min(1.0, trend * 10)
                reasons.append("APPROACHING_DEPTH_TREND")
            elif trend < -0.02:
                depth_score = 0.0
                reasons.append("RECEDING_DEPTH_TREND")

        # --- Intent score ---
        intent_score = 0.0
        if intent.state == IntentState.POSSIBLE_CROSSING:
            intent_score = 0.9
            reasons.append("POSSIBLE_CROSSING_INTENT")
        elif intent.state == IntentState.ENTERING_DANGER_ZONE:
            intent_score = 0.7
            reasons.append("ENTERING_VEHICLE_PATH")
        elif intent.state == IntentState.MOVING:
            intent_score = 0.3
        elif intent.state == IntentState.STATIONARY:
            intent_score = 0.1
        elif intent.state == IntentState.UNKNOWN:
            # Missing evidence should NOT be LOW risk
            intent_score = 0.4
            reasons.append("INSUFFICIENT_INTENT_DATA")

        # --- Confidence score ---
        conf_score = min(1.0, ped.confidence)

        # --- Weighted risk score ---
        risk_score = (
            self.config.weight_danger_zone * dz_score +
            self.config.weight_ttc * ttc_score +
            self.config.weight_depth_trend * depth_score +
            self.config.weight_intent * intent_score +
            self.config.weight_confidence * conf_score
        )

        # Clamp to [0, 1]
        risk_score = max(0.0, min(1.0, risk_score))
        assessment.risk_score = risk_score
        assessment.reason_codes = reasons

        # --- Classify risk level ---
        if risk_score >= self.config.high_risk_threshold:
            assessment.risk_level = RiskLevel.HIGH
        elif risk_score >= self.config.medium_risk_threshold:
            assessment.risk_level = RiskLevel.MEDIUM
        else:
            assessment.risk_level = RiskLevel.LOW

        # --- Apply hysteresis ---
        assessment = self._apply_hysteresis(tid, assessment, frame_index)

        return assessment

    def _estimate_ttc(
        self,
        ped: TrackedPedestrian,
        relative_depth: Optional[float],
        metric_distance: Optional[float]
    ) -> Tuple[Optional[float], TTCMethod, float]:
        """
        Estimate Time-to-Collision using available methods.

        Returns:
            (ttc_seconds, method, confidence)
            ttc_seconds is None if TTC cannot be reliably estimated.
        """
        # Method C: Calibrated metric TTC (preferred)
        metric_ttc = self._ttc_calibrated_metric(ped, metric_distance)
        if metric_ttc is not None:
            return metric_ttc

        # Method A: Bounding-box expansion
        bbox_ttc = self._ttc_bbox_expansion(ped)
        if bbox_ttc is not None:
            return bbox_ttc

        # Method B: Relative depth trend (least reliable)
        depth_ttc = self._ttc_depth_trend(ped, relative_depth)
        if depth_ttc is not None:
            return depth_ttc

        return (None, TTCMethod.UNAVAILABLE, 0.0)

    def _ttc_calibrated_metric(
        self,
        ped: TrackedPedestrian,
        metric_distance: Optional[float]
    ) -> Optional[Tuple[Optional[float], TTCMethod, float]]:
        """
        Method C: TTC from calibrated metric distance and closing speed.

        Only used when metric distance is available and closing speed
        can be estimated from temporal distance observations.
        """
        if metric_distance is None:
            return None

        # Need multiple metric distance observations
        # For now, we only have relative depth history, not metric history
        # This method is activated when set_calibration is used on depth estimator
        # and the system has accumulated enough metric observations

        return None  # Not yet available without metric history

    def _ttc_bbox_expansion(
        self,
        ped: TrackedPedestrian
    ) -> Optional[Tuple[Optional[float], TTCMethod, float]]:
        """
        Method A: TTC from bounding-box height expansion rate.

        LIMITATION: This is an IMAGE-PLANE approximation only.
        It should NOT be presented as calibrated physical TTC.
        Under a pinhole model, bbox height growth inversely relates
        to distance change, but accuracy depends on many assumptions.
        """
        heights = list(ped.history.bbox_heights)
        timestamps = list(ped.history.timestamps)

        if len(heights) < self.config.ttc_min_observations:
            return None
        if len(timestamps) < len(heights):
            return None

        # Use recent observations
        n = min(10, len(heights))
        recent_h = heights[-n:]
        recent_t = timestamps[-n:]

        dt = recent_t[-1] - recent_t[0]
        if dt < 0.1:
            return None

        dh = recent_h[-1] - recent_h[0]
        h_rate = dh / dt  # pixels/second

        # Only valid if bbox is growing (approaching)
        if h_rate <= 1.0:
            return None

        # Current height
        current_h = recent_h[-1]
        if current_h <= 10:
            return None

        # Approximate: TTC ~ current_height / growth_rate
        # This assumes constant approach speed (rough approximation)
        # A "full-frame" height would be ~frame_height, but we use a conservative
        # threshold to avoid overestimating
        target_h = current_h * 2.0  # Doubling as rough "very close" indicator
        ttc = (target_h - current_h) / h_rate

        if ttc < 0 or ttc > 30:
            return None

        # Confidence based on observation count and consistency
        confidence = min(0.6, n / 20.0)  # Cap at 0.6 (this method is approximate)

        return (ttc, TTCMethod.BBOX_EXPANSION, confidence)

    def _ttc_depth_trend(
        self,
        ped: TrackedPedestrian,
        relative_depth: Optional[float]
    ) -> Optional[Tuple[Optional[float], TTCMethod, float]]:
        """
        Method B: TTC from relative-depth trend.

        SEVERE LIMITATION: MiDaS relative depth values cannot be
        directly substituted into distance/speed TTC formulas.
        This method only produces a rough ordinal estimate.
        """
        depths = list(ped.history.depths)
        timestamps = list(ped.history.timestamps)

        if len(depths) < self.config.ttc_min_observations:
            return None
        if len(timestamps) < len(depths):
            return None

        n = min(10, len(depths))
        recent_d = depths[-n:]
        recent_t = timestamps[-n:]

        dt = recent_t[-1] - recent_t[0]
        if dt < 0.1:
            return None

        dd = recent_d[-1] - recent_d[0]
        d_rate = dd / dt  # relative depth change per second

        # Only valid if depth is increasing (approaching, in our normalization)
        if d_rate <= 0.005:
            return None

        current_d = recent_d[-1]
        if current_d <= 0.01:
            return None

        # Very rough estimate: how long until depth reaches 1.0 (very close)
        remaining = max(0.0, 1.0 - current_d)
        if d_rate > 0:
            ttc = remaining / d_rate
        else:
            return None

        if ttc < 0 or ttc > 30:
            return None

        # Low confidence — this is a relative-depth-based rough estimate
        confidence = min(0.4, n / 30.0)

        return (ttc, TTCMethod.DEPTH_TREND, confidence)

    def _apply_hysteresis(
        self,
        track_id: int,
        assessment: RiskAssessment,
        frame_index: int
    ) -> RiskAssessment:
        """
        Apply hysteresis to prevent rapid risk level flickering.

        A risk level change requires `hysteresis_frames` consecutive
        frames at the new level.
        """
        if track_id not in self._risk_history:
            self._risk_history[track_id] = deque(maxlen=self.config.risk_smoothing_window)

        history = self._risk_history[track_id]
        history.append(assessment.risk_level)

        prev_risk = self._previous_risk.get(track_id, RiskLevel.LOW)

        if assessment.risk_level != prev_risk:
            # Count consecutive frames at new level
            consecutive = 0
            for lvl in reversed(history):
                if lvl == assessment.risk_level:
                    consecutive += 1
                else:
                    break

            if consecutive < self.config.hysteresis_frames:
                # Not enough evidence — keep previous level
                assessment.risk_level = prev_risk
            else:
                # Transition confirmed
                self._previous_risk[track_id] = assessment.risk_level

        return assessment

    def should_alert(self, assessment: RiskAssessment) -> bool:
        """
        Determine if an alert should be sent for this assessment.

        Respects the configured cooldown period.

        Args:
            assessment: Risk assessment to check.

        Returns:
            True if an alert should be emitted.
        """
        if assessment.risk_level not in (RiskLevel.HIGH, RiskLevel.MEDIUM):
            return False

        now = time.time()
        last_alert = self._last_alert_time.get(assessment.track_id, 0)

        if now - last_alert < self.config.alert_cooldown_seconds:
            return False

        self._last_alert_time[assessment.track_id] = now
        return True

    def get_high_risk_assessments(
        self,
        assessments: List[RiskAssessment]
    ) -> List[RiskAssessment]:
        """Filter to only HIGH risk assessments."""
        return [a for a in assessments if a.risk_level == RiskLevel.HIGH]

    def get_alertable_assessments(
        self,
        assessments: List[RiskAssessment]
    ) -> List[RiskAssessment]:
        """Get assessments that should trigger alerts (respecting cooldown)."""
        return [a for a in assessments if self.should_alert(a)]
