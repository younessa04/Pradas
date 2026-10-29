"""
PRADAS Monocular Depth Estimation.

Uses MiDaS for relative depth estimation. Returns relative depth values,
NOT calibrated metric distances. Supports optional metric depth extension
when calibration data is available.

CRITICAL LIMITATION:
    MiDaS produces RELATIVE depth, not metric distance in meters.
    Never label relative depth values as meters without valid calibration.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from config.settings import DepthConfig
from utils.logging_config import get_logger

logger = get_logger("depth_estimator")


class MiDaSDepthEstimator:
    """
    Monocular relative depth estimator using MiDaS.

    Produces relative depth maps where higher values indicate closer objects
    (after inversion). Does NOT produce calibrated metric distance.

    Attributes:
        _model: MiDaS model instance.
        _transform: MiDaS input preprocessing transform.
        _initialized: Whether the model has been loaded.
        _calibration: Optional calibration parameters for metric conversion.
    """

    def __init__(self, config: Optional[DepthConfig] = None):
        """
        Args:
            config: Depth estimation configuration.
        """
        self.config = config or DepthConfig()
        self._model = None
        self._transform = None
        self._device = "cpu"
        self._initialized = False
        self._calibration: Optional[dict] = None  # For future metric extension

    def initialize(self, device: str = "cpu") -> bool:
        """
        Load the MiDaS model and preprocessing transform.

        Args:
            device: "cpu" or "cuda".

        Returns:
            True if model loaded successfully.
        """
        try:
            import torch

            self._device = device
            model_type = self.config.midas_model_type

            logger.info(f"Loading MiDaS model: {model_type} on {device}")

            # Load MiDaS via torch hub
            self._model = torch.hub.load(
                "intel-isl/MiDaS", model_type,
                trust_repo=True
            )
            self._model.to(device)
            self._model.eval()

            # Load the appropriate transform
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms",
                trust_repo=True
            )

            if model_type in ("DPT_Large", "DPT_Hybrid"):
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.small_transform

            self._initialized = True
            logger.info("MiDaS model loaded successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            self._initialized = False
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if the model is ready for inference."""
        return self._initialized and self._model is not None

    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate relative depth map for the given frame.

        Args:
            frame: BGR frame (OpenCV format).

        Returns:
            Depth map aligned to frame dimensions (float32),
            where HIGHER values = CLOSER to camera.
            None if estimation fails.
        """
        if not self.is_initialized:
            logger.warning("Depth estimator not initialized.")
            return None

        if frame is None or frame.size == 0:
            return None

        try:
            import torch

            h, w = frame.shape[:2]

            # Convert BGR to RGB for MiDaS
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply MiDaS transform
            input_batch = self._transform(frame_rgb).to(self._device)

            # Run inference without gradient tracking
            with torch.no_grad():
                prediction = self._model(input_batch)

                # Resize to original frame dimensions
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()

            # MiDaS output: larger values = farther away (inverse depth varies by model)
            # Normalize to [0, 1] range
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max - depth_min > 1e-6:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_map = np.zeros_like(depth_map)

            return depth_map.astype(np.float32)

        except Exception as e:
            logger.error(f"Depth estimation error: {e}")
            return None

    def get_pedestrian_depth(
        self,
        depth_map: np.ndarray,
        bbox: List[float],
        shrink_factor: Optional[float] = None
    ) -> Optional[float]:
        """
        Estimate the relative depth for a pedestrian given their bounding box.

        Uses the median of valid depth pixels within a shrunk ROI to
        avoid including background pixels.

        Args:
            depth_map: Depth map from estimate_depth().
            bbox: Bounding box [x1, y1, x2, y2].
            shrink_factor: Fraction to shrink the ROI inward (0-0.5).
                Defaults to config value.

        Returns:
            Relative depth value (0-1 range), where higher = closer.
            None if the ROI is invalid or empty.

        Note:
            This is a RELATIVE depth indicator. It does NOT represent
            calibrated metric distance (meters).
        """
        if depth_map is None or depth_map.size == 0:
            return None

        if len(bbox) < 4:
            return None

        h, w = depth_map.shape[:2]
        sf = shrink_factor if shrink_factor is not None else self.config.roi_shrink_factor

        x1, y1, x2, y2 = bbox[:4]

        # Shrink the ROI to focus on the body center
        bw = x2 - x1
        bh = y2 - y1
        x1_s = int(max(0, x1 + bw * sf))
        y1_s = int(max(0, y1 + bh * sf))
        x2_s = int(min(w, x2 - bw * sf))
        y2_s = int(min(h, y2 - bh * sf))

        if x2_s <= x1_s or y2_s <= y1_s:
            return None

        roi = depth_map[y1_s:y2_s, x1_s:x2_s]
        if roi.size == 0:
            return None

        # Use median for robustness against outliers
        median_depth = float(np.median(roi))
        return median_depth

    def depth_to_visualization(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert depth map to a colored visualization.

        Args:
            depth_map: Normalized depth map (0-1 float32).

        Returns:
            BGR colorized depth map (uint8).
        """
        if depth_map is None or depth_map.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Scale to 0-255
        depth_vis = (depth_map * 255).astype(np.uint8)
        # Apply colormap
        colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        return colored

    def set_calibration(
        self,
        scale: float,
        offset: float,
        unit: str = "meters"
    ) -> None:
        """
        Set calibration parameters for metric depth conversion.

        When calibration is set, get_metric_distance() can return
        approximate metric distances.

        Args:
            scale: Multiplicative scale factor.
            offset: Additive offset.
            unit: Distance unit (default: "meters").
        """
        self._calibration = {
            "scale": scale,
            "offset": offset,
            "unit": unit
        }
        logger.info(f"Depth calibration set: scale={scale}, offset={offset}, unit={unit}")

    def get_metric_distance(self, relative_depth: float) -> Optional[float]:
        """
        Convert relative depth to metric distance IF calibration is available.

        Args:
            relative_depth: Relative depth value from get_pedestrian_depth().

        Returns:
            Calibrated distance in configured units, or None if no
            calibration is available.

        Note:
            Returns None when no calibration exists. This method NEVER
            fabricates metric distances from uncalibrated relative depth.
        """
        if self._calibration is None:
            return None

        if relative_depth is None:
            return None

        try:
            distance = self._calibration["scale"] * relative_depth + self._calibration["offset"]
            return max(0.0, distance)
        except (KeyError, TypeError):
            return None
