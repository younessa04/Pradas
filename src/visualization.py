"""
PRADAS Frame Visualization.

Renders annotated frames with bounding boxes, skeletons, tracking IDs,
depth indicators, danger zone polygon, risk labels, and performance info.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.tracker import TrackedPedestrian
from src.detector_pose import SKELETON_CONNECTIONS, PedestrianDetection
from src.intent_analyzer import IntentResult, IntentState
from src.risk_ttc_engine import RiskAssessment, RiskLevel, DangerZoneStatus
from utils.logging_config import get_logger

logger = get_logger("visualization")

# Color scheme (BGR)
COLORS = {
    RiskLevel.LOW: (0, 200, 0),       # Green
    RiskLevel.MEDIUM: (0, 180, 255),   # Orange
    RiskLevel.HIGH: (0, 0, 255),       # Red
}

SKELETON_COLOR = (255, 200, 100)  # Light cyan
DANGER_ZONE_COLOR = (0, 100, 255)  # Orange-red
DANGER_ZONE_FILL_ALPHA = 0.15
INFO_COLOR = (255, 255, 255)  # White
INFO_BG_COLOR = (40, 40, 40)


class FrameVisualizer:
    """
    Renders annotated frames with detection, tracking, and risk information.

    Does NOT mutate model output tensors or the original frame.
    """

    def __init__(
        self,
        show_skeleton: bool = True,
        show_debug: bool = False,
        show_depth_indicator: bool = True,
        show_danger_zone: bool = True,
        font_scale: float = 0.5,
        thickness: int = 2
    ):
        """
        Args:
            show_skeleton: Whether to draw pose skeletons.
            show_debug: Whether to show detailed debug info.
            show_depth_indicator: Whether to show depth bar.
            show_danger_zone: Whether to draw the danger zone.
            font_scale: Font scale for text labels.
            thickness: Line thickness for bounding boxes.
        """
        self.show_skeleton = show_skeleton
        self.show_debug = show_debug
        self.show_depth_indicator = show_depth_indicator
        self.show_danger_zone = show_danger_zone
        self.font_scale = font_scale
        self.thickness = thickness
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        tracked_pedestrians: List[TrackedPedestrian],
        risk_assessments: List[RiskAssessment],
        intent_results: Dict[int, IntentResult],
        danger_zone_px: Optional[List[Tuple[int, int]]] = None,
        fps: float = 0.0,
        frame_index: int = 0,
        device: str = "cpu",
        frame_time_ms: float = 0.0
    ) -> np.ndarray:
        """
        Draw all annotations on a frame copy.

        Args:
            frame: Original BGR frame (not modified).
            tracked_pedestrians: Currently tracked pedestrians.
            risk_assessments: Risk assessments per pedestrian.
            intent_results: Intent analysis results.
            danger_zone_px: Danger zone polygon in pixel coords.
            fps: Current FPS.
            frame_index: Current frame index.
            device: Inference device name.
            frame_time_ms: Frame processing time in ms.

        Returns:
            Annotated frame copy.
        """
        if frame is None or frame.size == 0:
            return frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)

        vis = frame.copy()

        # Build lookup for risk assessments
        risk_map: Dict[int, RiskAssessment] = {
            a.track_id: a for a in risk_assessments
        }

        # 1. Draw danger zone
        if self.show_danger_zone and danger_zone_px and len(danger_zone_px) >= 3:
            self._draw_danger_zone(vis, danger_zone_px)

        # 2. Draw each tracked pedestrian
        for ped in tracked_pedestrians:
            risk = risk_map.get(ped.track_id)
            intent = intent_results.get(ped.track_id)
            self._draw_pedestrian(vis, ped, risk, intent)

        # 3. Draw HIGH risk alert banner
        high_risks = [a for a in risk_assessments if a.risk_level == RiskLevel.HIGH]
        if high_risks:
            self._draw_alert_banner(vis, len(high_risks))

        # 4. Draw HUD info
        self._draw_hud(vis, fps, frame_index, device, frame_time_ms,
                       len(tracked_pedestrians), len(high_risks))

        return vis

    def _draw_danger_zone(
        self,
        frame: np.ndarray,
        polygon: List[Tuple[int, int]]
    ) -> None:
        """Draw semi-transparent danger zone polygon."""
        pts = np.array(polygon, dtype=np.int32)

        # Semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], DANGER_ZONE_COLOR)
        cv2.addWeighted(overlay, DANGER_ZONE_FILL_ALPHA, frame,
                        1 - DANGER_ZONE_FILL_ALPHA, 0, frame)

        # Polygon outline
        cv2.polylines(frame, [pts], True, DANGER_ZONE_COLOR, 2)

        # Label
        centroid_x = int(np.mean(pts[:, 0]))
        centroid_y = int(np.mean(pts[:, 1]))
        cv2.putText(frame, "DANGER ZONE", (centroid_x - 50, centroid_y),
                    self._font, 0.5, DANGER_ZONE_COLOR, 1)

    def _draw_pedestrian(
        self,
        frame: np.ndarray,
        ped: TrackedPedestrian,
        risk: Optional[RiskAssessment],
        intent: Optional[IntentResult]
    ) -> None:
        """Draw all annotations for a single pedestrian."""
        bbox = ped.bbox
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Determine color from risk level
        risk_level = risk.risk_level if risk else RiskLevel.LOW
        color = COLORS.get(risk_level, (0, 200, 0))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

        # Label: Track ID + confidence
        label_parts = [f"ID:{ped.track_id}", f"{ped.confidence:.2f}"]

        if risk:
            label_parts.append(risk.risk_level.value)
            if risk.ttc_seconds is not None and risk.ttc_confidence > 0.3:
                label_parts.append(f"TTC:{risk.ttc_seconds:.1f}s*")

        label = " | ".join(label_parts)
        label_y = max(y1 - 8, 15)

        # Label background
        (text_w, text_h), _ = cv2.getTextSize(label, self._font, self.font_scale, 1)
        cv2.rectangle(frame, (x1, label_y - text_h - 4),
                      (x1 + text_w + 4, label_y + 4), INFO_BG_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, label_y),
                    self._font, self.font_scale, color, 1)

        # Skeleton
        if self.show_skeleton and ped.keypoints is not None:
            self._draw_skeleton(frame, ped.keypoints, color)

        # Depth indicator bar (on right side of bbox)
        if self.show_depth_indicator and risk and risk.relative_depth is not None:
            self._draw_depth_bar(frame, x2 + 4, y1, y2, risk.relative_depth)

        # Intent state label (below bbox)
        if intent and self.show_debug:
            intent_label = f"{intent.state.value} spd:{intent.speed:.1f}"
            cv2.putText(frame, intent_label, (x1, y2 + 15),
                        self._font, 0.4, color, 1)

        # Movement direction arrow
        if intent and intent.speed > 3.0:
            cx, cy = int(ped.center[0]), int(ped.center[1])
            arrow_len = min(40, int(intent.speed * 3))
            import math
            end_x = cx + int(arrow_len * math.cos(math.radians(intent.direction_degrees)))
            end_y = cy + int(arrow_len * math.sin(math.radians(intent.direction_degrees)))
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        color: Tuple[int, int, int]
    ) -> None:
        """Draw pose skeleton connections and keypoint dots."""
        if keypoints is None or len(keypoints) == 0:
            return

        # Draw connections
        for i, j in SKELETON_CONNECTIONS:
            if i >= len(keypoints) or j >= len(keypoints):
                continue
            kp1 = keypoints[i]
            kp2 = keypoints[j]

            # Check confidence
            conf1 = kp1[2] if len(kp1) > 2 else 1.0
            conf2 = kp2[2] if len(kp2) > 2 else 1.0

            if conf1 < 0.3 or conf2 < 0.3:
                continue

            pt1 = (int(kp1[0]), int(kp1[1]))
            pt2 = (int(kp2[0]), int(kp2[1]))

            if pt1[0] <= 0 and pt1[1] <= 0:
                continue
            if pt2[0] <= 0 and pt2[1] <= 0:
                continue

            cv2.line(frame, pt1, pt2, SKELETON_COLOR, 1)

        # Draw keypoints
        for kp in keypoints:
            conf = kp[2] if len(kp) > 2 else 1.0
            if conf < 0.3:
                continue
            pt = (int(kp[0]), int(kp[1]))
            if pt[0] <= 0 and pt[1] <= 0:
                continue
            cv2.circle(frame, pt, 3, color, -1)

    def _draw_depth_bar(
        self,
        frame: np.ndarray,
        x: int, y1: int, y2: int,
        relative_depth: float
    ) -> None:
        """Draw a vertical depth indicator bar."""
        bar_w = 8
        bar_h = y2 - y1
        if bar_h <= 0:
            return

        fill_h = int(bar_h * max(0.0, min(1.0, relative_depth)))

        # Background
        cv2.rectangle(frame, (x, y1), (x + bar_w, y2), (80, 80, 80), -1)

        # Fill (green=far, red=close)
        fill_color = (
            int(255 * relative_depth),
            0,
            int(255 * (1 - relative_depth))
        )
        cv2.rectangle(frame, (x, y2 - fill_h), (x + bar_w, y2), fill_color, -1)

        # Border
        cv2.rectangle(frame, (x, y1), (x + bar_w, y2), (150, 150, 150), 1)

    def _draw_alert_banner(self, frame: np.ndarray, count: int) -> None:
        """Draw a HIGH RISK alert banner at the top of the frame."""
        h, w = frame.shape[:2]
        banner_h = 40

        # Red semi-transparent banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        text = f"⚠ HIGH RISK — {count} pedestrian(s) in danger"
        cv2.putText(frame, text, (w // 2 - 200, 28),
                    self._font, 0.7, (255, 255, 255), 2)

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        frame_index: int,
        device: str,
        frame_time_ms: float,
        pedestrian_count: int,
        high_risk_count: int
    ) -> None:
        """Draw heads-up display with performance and status info."""
        h, w = frame.shape[:2]
        y_offset = h - 80

        # Background panel
        cv2.rectangle(frame, (5, y_offset), (300, h - 5), INFO_BG_COLOR, -1)
        cv2.rectangle(frame, (5, y_offset), (300, h - 5), (100, 100, 100), 1)

        lines = [
            f"FPS: {fps:.1f} | Frame: {frame_index} | {frame_time_ms:.0f}ms",
            f"Device: {device} | Peds: {pedestrian_count} | Alerts: {high_risk_count}",
            "PRADAS v1.0 — Research Prototype"
        ]

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, y_offset + 18 + i * 20),
                        self._font, 0.4, INFO_COLOR, 1)
