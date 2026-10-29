"""
Tests for PRADAS Monocular Depth Estimator.

Focuses on ROI extraction, median depth calculation, calibration,
metric conversion, and visualization without requiring model weights.
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import DepthConfig
from src.depth_estimator import MiDaSDepthEstimator


@pytest.fixture
def depth_estimator():
    """Create a DepthEstimator without downloading weights."""
    config = DepthConfig(roi_shrink_factor=0.2)
    return MiDaSDepthEstimator(config)


@pytest.fixture
def synthetic_depth_map():
    """Create a synthetic depth map (gradient from far to near)."""
    # 480x640 depth map where values range from 0.0 (far) to 1.0 (near)
    y, x = np.mgrid[0:480, 0:640]
    depth_map = (y / 480.0).astype(np.float32)
    return depth_map


class TestDepthEstimatorUnit:
    """Unit tests for MiDaSDepthEstimator methods."""

    def test_uninitialized_returns_none(self, depth_estimator):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert not depth_estimator.is_initialized
        assert depth_estimator.estimate_depth(frame) is None

    def test_get_pedestrian_depth_valid_bbox(self, depth_estimator, synthetic_depth_map):
        # Pedestrian in lower half (y in 240..400) -> depth roughly 0.5 to 0.83
        bbox = [100, 240, 200, 400]
        depth = depth_estimator.get_pedestrian_depth(synthetic_depth_map, bbox)

        assert depth is not None
        assert 0.0 <= depth <= 1.0
        # Median should be roughly around center y = 320 / 480 = 0.667
        assert 0.55 <= depth <= 0.75

    def test_get_pedestrian_depth_invalid_bbox(self, depth_estimator, synthetic_depth_map):
        # Empty or inverted bbox
        assert depth_estimator.get_pedestrian_depth(synthetic_depth_map, [200, 200, 100, 100]) is None
        assert depth_estimator.get_pedestrian_depth(synthetic_depth_map, [100, 100]) is None

    def test_get_pedestrian_depth_out_of_bounds(self, depth_estimator, synthetic_depth_map):
        # Bbox entirely outside frame
        assert depth_estimator.get_pedestrian_depth(synthetic_depth_map, [700, 700, 800, 800]) is None

    def test_get_pedestrian_depth_empty_depth_map(self, depth_estimator):
        empty_map = np.empty((0, 0), dtype=np.float32)
        assert depth_estimator.get_pedestrian_depth(empty_map, [10, 10, 50, 50]) is None
        assert depth_estimator.get_pedestrian_depth(None, [10, 10, 50, 50]) is None

    def test_depth_to_visualization(self, depth_estimator, synthetic_depth_map):
        vis = depth_estimator.depth_to_visualization(synthetic_depth_map)
        assert vis.shape == (480, 640, 3)
        assert vis.dtype == np.uint8

    def test_depth_to_visualization_empty(self, depth_estimator):
        vis = depth_estimator.depth_to_visualization(None)
        assert vis is not None
        assert vis.shape == (100, 100, 3)

    def test_metric_distance_without_calibration(self, depth_estimator):
        # When calibration is not set, metric distance MUST be None
        assert depth_estimator.get_metric_distance(0.5) is None
        assert depth_estimator.get_metric_distance(None) is None

    def test_metric_distance_with_calibration(self, depth_estimator):
        depth_estimator.set_calibration(scale=10.0, offset=1.0, unit="meters")
        # 10.0 * 0.5 + 1.0 = 6.0
        metric = depth_estimator.get_metric_distance(0.5)
        assert metric is not None
        assert pytest.approx(metric, 0.01) == 6.0

    def test_metric_distance_clamped_at_zero(self, depth_estimator):
        depth_estimator.set_calibration(scale=-10.0, offset=-2.0, unit="meters")
        metric = depth_estimator.get_metric_distance(0.5)
        assert metric == 0.0
