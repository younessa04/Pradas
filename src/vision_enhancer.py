"""
PRADAS Vision Enhancer — Low-Light / Nighttime Video Enhancement.

Enhances dark or nighttime frames using gamma correction and CLAHE.
Includes automatic low-light detection based on average luminance.
"""

import cv2
import numpy as np
from typing import Optional

from config.settings import NightEnhancementConfig
from utils.logging_config import get_logger

logger = get_logger("vision_enhancer")


class VisionEnhancer:
    """
    Enhances low-light / nighttime video frames for improved detection.

    Supports:
        - Gamma correction (brightens dark regions).
        - CLAHE (Contrast Limited Adaptive Histogram Equalization).
        - Combined mode applying both techniques.
        - Automatic detection of low-light conditions.
    """

    def __init__(self, config: Optional[NightEnhancementConfig] = None):
        """
        Args:
            config: Enhancement configuration. Uses defaults if None.
        """
        self.config = config or NightEnhancementConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )
        # Precompute gamma lookup table
        self._gamma_lut = self._build_gamma_lut(self.config.gamma)
        logger.info(f"VisionEnhancer initialized: mode={self.config.mode}, "
                    f"gamma={self.config.gamma}, "
                    f"clahe_clip={self.config.clahe_clip_limit}")

    def _build_gamma_lut(self, gamma: float) -> np.ndarray:
        """Build lookup table for gamma correction (gamma < 1.0 brightens dark regions)."""
        gamma_val = max(gamma, 0.01)
        table = np.array([
            ((i / 255.0) ** gamma_val) * 255
            for i in range(256)
        ], dtype=np.uint8)
        return table

    def _is_low_light(self, frame: np.ndarray) -> bool:
        """
        Detect if the frame is low-light based on average luminance.

        Args:
            frame: BGR frame.

        Returns:
            True if average luminance is below threshold.
        """
        if frame is None or frame.size == 0:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_luminance = float(np.mean(gray))
        return avg_luminance < self.config.luminance_threshold

    def _apply_gamma(self, frame: np.ndarray) -> np.ndarray:
        """Apply gamma correction using precomputed LUT."""
        return cv2.LUT(frame, self._gamma_lut)

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE on the L channel of LAB color space.

        This preserves color while enhancing contrast.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance a video frame if low-light conditions are detected.

        Args:
            frame: Input BGR frame (OpenCV format).

        Returns:
            Enhanced frame (same dimensions and dtype as input),
            or the original frame if enhancement is not needed/disabled.
        """
        # Handle invalid input
        if frame is None or frame.size == 0:
            logger.warning("Empty or None frame passed to enhance().")
            return frame if frame is not None else np.empty((0, 0, 3), dtype=np.uint8)

        mode = self.config.mode.lower()

        # Check if enhancement is disabled
        if mode == "never":
            return frame

        # Auto-detect low light
        if mode == "auto":
            if not self._is_low_light(frame):
                return frame

        # Apply combined enhancement: gamma first, then CLAHE
        try:
            enhanced = self._apply_gamma(frame)
            enhanced = self._apply_clahe(enhanced)
            return enhanced
        except cv2.error as e:
            logger.error(f"OpenCV error during enhancement: {e}")
            return frame

    def get_luminance(self, frame: np.ndarray) -> float:
        """
        Get average luminance of a frame.

        Args:
            frame: BGR frame.

        Returns:
            Average luminance value (0-255). Returns 0 for invalid frames.
        """
        if frame is None or frame.size == 0:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
