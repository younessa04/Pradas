"""
PRADAS Pedestrian Detection and Pose Estimation.

Uses Ultralytics YOLOv8 for person detection with skeletal keypoint extraction.
Returns structured detection results for downstream tracking and analysis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config.settings import DetectionConfig
from utils.logging_config import get_logger

logger = get_logger("detector_pose")


# COCO keypoint indices
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# COCO skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head
    (5, 6),                                  # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),        # Arms
    (5, 11), (6, 12),                        # Torso
    (11, 12),                                # Hips
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]


@dataclass
class PedestrianDetection:
    """
    Structured detection result for a single pedestrian.

    Attributes:
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates.
        confidence: Detection confidence score [0, 1].
        class_id: Object class ID (0 = person in COCO).
        keypoints: Nx3 array of (x, y, confidence) for each keypoint.
            None if keypoints are not available.
        frame_index: Source frame index.
    """
    bbox: List[float]
    confidence: float
    class_id: int = 0
    keypoints: Optional[np.ndarray] = None
    frame_index: int = 0

    @property
    def center(self) -> Tuple[float, float]:
        """Bounding box center (x, y)."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0
        )

    @property
    def height(self) -> float:
        """Bounding box height."""
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def width(self) -> float:
        """Bounding box width."""
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def area(self) -> float:
        """Bounding box area."""
        return self.width * self.height

    def get_keypoint(self, name: str) -> Optional[Tuple[float, float, float]]:
        """
        Get a specific keypoint by name.

        Args:
            name: Keypoint name (e.g., "nose", "left_shoulder").

        Returns:
            (x, y, confidence) or None if not available.
        """
        if self.keypoints is None:
            return None
        if name not in KEYPOINT_NAMES:
            return None
        idx = KEYPOINT_NAMES.index(name)
        if idx < len(self.keypoints):
            kp = self.keypoints[idx]
            return (float(kp[0]), float(kp[1]), float(kp[2]))
        return None

    def get_shoulder_center(self) -> Optional[Tuple[float, float]]:
        """Get center point between shoulders if both visible."""
        ls = self.get_keypoint("left_shoulder")
        rs = self.get_keypoint("right_shoulder")
        if ls and rs and ls[2] > 0.3 and rs[2] > 0.3:
            return ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        return None

    def get_hip_center(self) -> Optional[Tuple[float, float]]:
        """Get center point between hips if both visible."""
        lh = self.get_keypoint("left_hip")
        rh = self.get_keypoint("right_hip")
        if lh and rh and lh[2] > 0.3 and rh[2] > 0.3:
            return ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
        return None

    def estimate_torso_orientation(self) -> Optional[str]:
        """
        Estimate torso orientation from shoulder keypoints.

        This is an IMAGE-PLANE estimate only. It does not reliably indicate
        real-world body facing direction without camera calibration.

        Returns:
            "facing_left", "facing_right", "facing_camera", or None.
        """
        ls = self.get_keypoint("left_shoulder")
        rs = self.get_keypoint("right_shoulder")
        if not ls or not rs or ls[2] < 0.3 or rs[2] < 0.3:
            return None

        shoulder_width = abs(rs[0] - ls[0])
        shoulder_height_diff = abs(rs[1] - ls[1])

        # If shoulders are very narrow, person may be facing sideways
        bbox_w = self.width
        if bbox_w > 0 and shoulder_width / bbox_w < 0.15:
            if ls[0] < rs[0]:
                return "facing_right"
            return "facing_left"

        # If shoulders are roughly symmetric, person may face camera
        if bbox_w > 0 and shoulder_width / bbox_w > 0.3:
            return "facing_camera"

        return None


class PedestrianDetectorPose:
    """
    Detects pedestrians and extracts pose keypoints using YOLOv8.

    The model is loaded once at initialization to avoid per-frame overhead.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Args:
            config: Detection configuration. Uses defaults if None.
        """
        self.config = config or DetectionConfig()
        self._model = None
        self._device = "cpu"
        self._initialized = False

    def initialize(self, device: str = "cpu") -> bool:
        """
        Load the YOLOv8 pose model.

        Args:
            device: "cpu" or "cuda".

        Returns:
            True if model loaded successfully.
        """
        try:
            from ultralytics import YOLO

            self._device = device
            logger.info(f"Loading YOLO pose model: {self.config.yolo_pose_model} on {device}")
            self._model = YOLO(self.config.yolo_pose_model)
            self._initialized = True
            logger.info("YOLO pose model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._initialized = False
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if model is ready for inference."""
        return self._initialized and self._model is not None

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = 0
    ) -> List[PedestrianDetection]:
        """
        Detect pedestrians and extract pose keypoints from a frame.

        Args:
            frame: BGR frame (OpenCV format).
            frame_index: Current frame index for tracking context.

        Returns:
            List of PedestrianDetection objects. Empty list if no detections
            or model not initialized.
        """
        if not self.is_initialized:
            logger.warning("Detector not initialized. Call initialize() first.")
            return []

        if frame is None or frame.size == 0:
            return []

        detections: List[PedestrianDetection] = []

        try:
            results = self._model(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                device=self._device,
                max_det=self.config.max_detections,
                imgsz=self.config.imgsz,
                verbose=False,
                classes=[self.config.person_class_id]
            )

            if not results or len(results) == 0:
                return []

            result = results[0]

            # Extract boxes
            if result.boxes is None or len(result.boxes) == 0:
                return []

            boxes = result.boxes
            n_detections = len(boxes)

            # Extract keypoints if available
            has_keypoints = (
                hasattr(result, 'keypoints') and
                result.keypoints is not None and
                result.keypoints.data is not None and
                len(result.keypoints.data) > 0
            )

            for i in range(n_detections):
                # Bounding box
                xyxy = boxes.xyxy[i].cpu().numpy()
                bbox = [float(xyxy[0]), float(xyxy[1]),
                        float(xyxy[2]), float(xyxy[3])]

                # Confidence
                conf = float(boxes.conf[i].cpu().numpy())

                # Class ID
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Only keep person detections
                if cls_id != self.config.person_class_id:
                    continue

                # Validate bounding box
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue

                # Keypoints
                kps = None
                if has_keypoints and i < len(result.keypoints.data):
                    kp_data = result.keypoints.data[i].cpu().numpy()
                    # kp_data shape: (17, 3) for COCO — x, y, confidence
                    if kp_data.ndim == 2 and kp_data.shape[-1] >= 2:
                        # Ensure 3 columns (x, y, conf)
                        if kp_data.shape[-1] == 2:
                            conf_col = np.ones((kp_data.shape[0], 1))
                            kp_data = np.hstack([kp_data, conf_col])
                        kps = kp_data

                detections.append(PedestrianDetection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    keypoints=kps,
                    frame_index=frame_index
                ))

        except Exception as e:
            logger.error(f"Detection error at frame {frame_index}: {e}")

        return detections

    def detect_raw(self, frame: np.ndarray) -> object:
        """
        Run raw YOLO inference and return the Ultralytics Results object.

        Useful for the tracker which needs the raw results for ByteTrack.

        Args:
            frame: BGR frame.

        Returns:
            Ultralytics Results object, or None on error.
        """
        if not self.is_initialized or frame is None or frame.size == 0:
            return None
        try:
            results = self._model(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                device=self._device,
                max_det=self.config.max_detections,
                imgsz=self.config.imgsz,
                verbose=False,
                classes=[self.config.person_class_id]
            )
            return results
        except Exception as e:
            logger.error(f"Raw detection error: {e}")
            return None
