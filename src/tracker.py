"""
PRADAS Multi-Object Pedestrian Tracker.

Maintains persistent pedestrian IDs across frames using Ultralytics'
ByteTrack integration. Stores bounded history for temporal analysis.
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.settings import TrackingConfig
from src.detector_pose import PedestrianDetection
from utils.logging_config import get_logger

logger = get_logger("tracker")


@dataclass
class TrackHistory:
    """Bounded history for a single tracked pedestrian."""
    centers: deque = field(default_factory=lambda: deque(maxlen=90))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=90))
    bbox_heights: deque = field(default_factory=lambda: deque(maxlen=90))
    bbox_areas: deque = field(default_factory=lambda: deque(maxlen=90))
    depths: deque = field(default_factory=lambda: deque(maxlen=90))
    frame_indices: deque = field(default_factory=lambda: deque(maxlen=90))


@dataclass
class TrackedPedestrian:
    """
    A tracked pedestrian with persistent identity and history.

    Attributes:
        track_id: Persistent tracking ID.
        bbox: Current bounding box [x1, y1, x2, y2].
        confidence: Current detection confidence.
        keypoints: Current keypoints (Nx3 array or None).
        frame_index: Last seen frame index.
        center: Current center (x, y).
        track_age: Number of frames this track has existed.
        last_seen_frame: Frame index when last detected.
        first_seen_frame: Frame index when first created.
        history: Bounded history of positions, sizes, depths.
    """
    track_id: int
    bbox: List[float]
    confidence: float
    keypoints: Optional[np.ndarray] = None
    frame_index: int = 0
    track_age: int = 1
    last_seen_frame: int = 0
    first_seen_frame: int = 0
    history: TrackHistory = field(default_factory=TrackHistory)

    @property
    def center(self) -> Tuple[float, float]:
        """Current bounding box center."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0
        )

    @property
    def height(self) -> float:
        """Current bounding box height."""
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        """Current bounding box area."""
        w = max(0.0, self.bbox[2] - self.bbox[0])
        h = max(0.0, self.bbox[3] - self.bbox[1])
        return w * h

    def update(
        self,
        bbox: List[float],
        confidence: float,
        keypoints: Optional[np.ndarray],
        frame_index: int,
        depth: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Update track with new observation.

        Args:
            bbox: New bounding box.
            confidence: New detection confidence.
            keypoints: New keypoints.
            frame_index: Current frame index.
            depth: Optional relative depth value.
            timestamp: Optional timestamp.
        """
        self.bbox = bbox
        self.confidence = confidence
        self.keypoints = keypoints
        self.frame_index = frame_index
        self.last_seen_frame = frame_index
        self.track_age += 1

        ts = timestamp or time.time()
        cx, cy = self.center

        self.history.centers.append((cx, cy))
        self.history.timestamps.append(ts)
        self.history.bbox_heights.append(self.height)
        self.history.bbox_areas.append(self.area)
        self.history.frame_indices.append(frame_index)

        if depth is not None:
            self.history.depths.append(depth)


class PedestrianTracker:
    """
    Multi-object pedestrian tracker.

    Uses Ultralytics' built-in ByteTrack tracker when possible.
    Falls back to a simple IoU-based tracker if the YOLO tracking
    interface is not available.
    """

    def __init__(
        self,
        config: Optional[TrackingConfig] = None,
        device: str = "cpu",
        imgsz: int = 640
    ):
        """
        Args:
            config: Tracking configuration. Uses defaults if None.
            device: Inference device ("cpu" or "cuda").
            imgsz: Inference input resolution for YOLO tracker.
        """
        self.config = config or TrackingConfig()
        self._tracks: Dict[int, TrackedPedestrian] = {}
        self._frame_count: int = 0
        self._use_yolo_track: bool = True
        self._model = None
        self._device = device
        self._imgsz = imgsz
        logger.info(f"PedestrianTracker initialized: imgsz={imgsz}, device={device}, "
                    f"conf={self.config.track_high_thresh}")

    def set_model(self, model) -> None:
        """
        Set the YOLO model reference for built-in tracking.

        Args:
            model: Ultralytics YOLO model instance.
        """
        self._model = model

    def set_device_and_imgsz(self, device: str, imgsz: int) -> None:
        """Update tracker device and inference image size."""
        self._device = device
        self._imgsz = imgsz

    def update(
        self,
        frame: np.ndarray,
        detections: List[PedestrianDetection],
        frame_index: int,
        timestamp: Optional[float] = None
    ) -> List[TrackedPedestrian]:
        """
        Update tracks with new frame detections.

        Tries YOLO's built-in ByteTrack first. Falls back to simple
        assignment if tracking API is unavailable.

        Args:
            frame: Current BGR frame.
            detections: List of detections from the detector.
            frame_index: Current frame index.
            timestamp: Optional frame timestamp.

        Returns:
            List of currently active TrackedPedestrian objects.
        """
        self._frame_count = frame_index

        if self._use_yolo_track and self._model is not None:
            return self._update_with_yolo_track(frame, detections, frame_index, timestamp)
        else:
            return self._update_simple(detections, frame_index, timestamp)

    def _update_with_yolo_track(
        self,
        frame: np.ndarray,
        detections: List[PedestrianDetection],
        frame_index: int,
        timestamp: Optional[float] = None
    ) -> List[TrackedPedestrian]:
        """Use Ultralytics' built-in tracker."""
        try:
            from ultralytics import YOLO

            results = self._model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.config.track_high_thresh,
                iou=self.config.match_thresh,
                imgsz=self._imgsz,
                device=self._device,
                verbose=False,
                classes=[0]  # Person only
            )

            if not results or len(results) == 0:
                return self._get_active_tracks(frame_index)

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return self._get_active_tracks(frame_index)

            boxes = result.boxes
            seen_ids = set()

            # Extract keypoints if available
            has_kps = (
                hasattr(result, 'keypoints') and
                result.keypoints is not None and
                result.keypoints.data is not None and
                len(result.keypoints.data) > 0
            )

            for i in range(len(boxes)):
                # Get track ID
                if boxes.id is None:
                    continue
                track_id = int(boxes.id[i].cpu().numpy())
                seen_ids.add(track_id)

                # Bounding box
                xyxy = boxes.xyxy[i].cpu().numpy()
                bbox = [float(xyxy[0]), float(xyxy[1]),
                        float(xyxy[2]), float(xyxy[3])]
                conf = float(boxes.conf[i].cpu().numpy())

                # Keypoints
                kps = None
                if has_kps and i < len(result.keypoints.data):
                    kp_data = result.keypoints.data[i].cpu().numpy()
                    if kp_data.ndim == 2 and kp_data.shape[-1] >= 2:
                        if kp_data.shape[-1] == 2:
                            conf_col = np.ones((kp_data.shape[0], 1))
                            kp_data = np.hstack([kp_data, conf_col])
                        kps = kp_data

                # Update or create track
                if track_id in self._tracks:
                    self._tracks[track_id].update(
                        bbox=bbox, confidence=conf, keypoints=kps,
                        frame_index=frame_index, timestamp=timestamp
                    )
                else:
                    tp = TrackedPedestrian(
                        track_id=track_id,
                        bbox=bbox,
                        confidence=conf,
                        keypoints=kps,
                        frame_index=frame_index,
                        track_age=1,
                        last_seen_frame=frame_index,
                        first_seen_frame=frame_index
                    )
                    ts = timestamp or time.time()
                    tp.history.centers.append(tp.center)
                    tp.history.timestamps.append(ts)
                    tp.history.bbox_heights.append(tp.height)
                    tp.history.bbox_areas.append(tp.area)
                    tp.history.frame_indices.append(frame_index)
                    self._tracks[track_id] = tp

            # Expire old tracks
            self._expire_tracks(frame_index)

            return self._get_active_tracks(frame_index)

        except Exception as e:
            logger.warning(f"YOLO tracking failed, falling back to simple tracker: {e}")
            self._use_yolo_track = False
            return self._update_simple(detections, frame_index, timestamp)

    def _update_simple(
        self,
        detections: List[PedestrianDetection],
        frame_index: int,
        timestamp: Optional[float] = None
    ) -> List[TrackedPedestrian]:
        """
        Simple IoU-based tracking fallback.

        Assigns detections to existing tracks by IoU matching.
        Creates new tracks for unmatched detections.
        """
        if not detections:
            self._expire_tracks(frame_index)
            return self._get_active_tracks(frame_index)

        active = {tid: t for tid, t in self._tracks.items()
                  if frame_index - t.last_seen_frame <= self.config.track_buffer}

        matched_tracks = set()
        matched_dets = set()

        # Match detections to existing tracks by IoU
        for det_idx, det in enumerate(detections):
            best_iou = 0.3  # Minimum IoU threshold
            best_tid = None

            for tid, track in active.items():
                if tid in matched_tracks:
                    continue
                iou = self._compute_iou(det.bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                self._tracks[best_tid].update(
                    bbox=det.bbox, confidence=det.confidence,
                    keypoints=det.keypoints, frame_index=frame_index,
                    timestamp=timestamp
                )
                matched_tracks.add(best_tid)
                matched_dets.add(det_idx)

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in matched_dets:
                continue
            new_id = self._next_track_id()
            tp = TrackedPedestrian(
                track_id=new_id,
                bbox=det.bbox,
                confidence=det.confidence,
                keypoints=det.keypoints,
                frame_index=frame_index,
                track_age=1,
                last_seen_frame=frame_index,
                first_seen_frame=frame_index
            )
            ts = timestamp or time.time()
            tp.history.centers.append(tp.center)
            tp.history.timestamps.append(ts)
            tp.history.bbox_heights.append(tp.height)
            tp.history.bbox_areas.append(tp.area)
            tp.history.frame_indices.append(frame_index)
            self._tracks[new_id] = tp

        self._expire_tracks(frame_index)
        return self._get_active_tracks(frame_index)

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _next_track_id(self) -> int:
        """Generate next track ID."""
        if not self._tracks:
            return 1
        return max(self._tracks.keys()) + 1

    def _expire_tracks(self, current_frame: int) -> None:
        """Remove tracks that haven't been seen for too long."""
        expired = [
            tid for tid, track in self._tracks.items()
            if current_frame - track.last_seen_frame > self.config.track_buffer
        ]
        for tid in expired:
            del self._tracks[tid]
            logger.debug(f"Track {tid} expired at frame {current_frame}")

    def _get_active_tracks(self, current_frame: int) -> List[TrackedPedestrian]:
        """Get tracks seen within the track buffer."""
        return [
            track for track in self._tracks.values()
            if current_frame - track.last_seen_frame <= self.config.track_buffer
        ]

    def get_track(self, track_id: int) -> Optional[TrackedPedestrian]:
        """Get a specific track by ID."""
        return self._tracks.get(track_id)

    @property
    def active_track_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._tracks)

    def reset(self) -> None:
        """
        Reset all tracks.

        WARNING: Track IDs are not guaranteed to be globally unique after reset.
        """
        self._tracks.clear()
        logger.info("Tracker reset. All tracks cleared.")
