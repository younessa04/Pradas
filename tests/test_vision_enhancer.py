"""
Tests for PRADAS Vision Enhancer.
"""

import pytest
import numpy as np
import cv2

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import NightEnhancementConfig
from src.vision_enhancer import VisionEnhancer


@pytest.fixture
def enhancer():
    """Create a VisionEnhancer with default config."""
    config = NightEnhancementConfig(mode="always")
    return VisionEnhancer(config)


@pytest.fixture
def enhancer_auto():
    """Create a VisionEnhancer in auto mode."""
    config = NightEnhancementConfig(mode="auto", luminance_threshold=80.0)
    return VisionEnhancer(config)


@pytest.fixture
def enhancer_disabled():
    """Create a disabled VisionEnhancer."""
    config = NightEnhancementConfig(mode="never")
    return VisionEnhancer(config)


@pytest.fixture
def dark_frame():
    """Create a synthetic dark frame (low luminance)."""
    frame = np.full((480, 640, 3), 20, dtype=np.uint8)
    return frame


@pytest.fixture
def bright_frame():
    """Create a synthetic bright frame."""
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    return frame


@pytest.fixture
def normal_frame():
    """Create a normal test frame."""
    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    return frame


class TestVisionEnhancer:
    """Tests for the VisionEnhancer class."""

    def test_enhance_preserves_dimensions(self, enhancer, dark_frame):
        result = enhancer.enhance(dark_frame)
        assert result.shape == dark_frame.shape

    def test_enhance_preserves_dtype(self, enhancer, dark_frame):
        result = enhancer.enhance(dark_frame)
        assert result.dtype == np.uint8

    def test_enhance_empty_frame(self, enhancer):
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        result = enhancer.enhance(empty)
        assert result is not None

    def test_enhance_none_frame(self, enhancer):
        result = enhancer.enhance(None)
        assert result is not None

    def test_enhance_brightens_dark_frame(self, enhancer, dark_frame):
        result = enhancer.enhance(dark_frame)
        # Enhanced frame should be brighter
        assert np.mean(result) > np.mean(dark_frame)

    def test_disabled_returns_original(self, enhancer_disabled, dark_frame):
        result = enhancer_disabled.enhance(dark_frame)
        np.testing.assert_array_equal(result, dark_frame)

    def test_auto_mode_enhances_dark(self, enhancer_auto, dark_frame):
        result = enhancer_auto.enhance(dark_frame)
        # Should enhance dark frames
        assert np.mean(result) > np.mean(dark_frame)

    def test_auto_mode_skips_bright(self, enhancer_auto, bright_frame):
        result = enhancer_auto.enhance(bright_frame)
        # Should not enhance bright frames (returns original)
        np.testing.assert_array_equal(result, bright_frame)

    def test_get_luminance_dark(self, enhancer, dark_frame):
        lum = enhancer.get_luminance(dark_frame)
        assert lum < 50

    def test_get_luminance_bright(self, enhancer, bright_frame):
        lum = enhancer.get_luminance(bright_frame)
        assert lum > 100

    def test_get_luminance_none(self, enhancer):
        assert enhancer.get_luminance(None) == 0.0

    def test_get_luminance_empty(self, enhancer):
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        assert enhancer.get_luminance(empty) == 0.0

    def test_enhance_normal_frame_always_mode(self, enhancer, normal_frame):
        result = enhancer.enhance(normal_frame)
        assert result.shape == normal_frame.shape
        assert result.dtype == np.uint8

    def test_gamma_correction_custom(self):
        config = NightEnhancementConfig(mode="always", gamma=0.3)
        enh = VisionEnhancer(config)
        dark = np.full((100, 100, 3), 30, dtype=np.uint8)
        result = enh.enhance(dark)
        assert np.mean(result) > np.mean(dark)
