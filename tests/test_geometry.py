"""
Tests for PRADAS Geometry Utilities.
"""

import pytest
import math

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.geometry import (
    point_in_polygon,
    bbox_center,
    bbox_area,
    bbox_height,
    bbox_width,
    bbox_bottom_center,
    normalize_point,
    denormalize_polygon,
    compute_velocity,
    distance_to_polygon_edge,
)


# --- Point in Polygon ---

class TestPointInPolygon:
    """Tests for point_in_polygon function."""

    def test_point_inside_square(self):
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((5, 5), polygon) is True

    def test_point_outside_square(self):
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((15, 5), polygon) is False

    def test_point_above_square(self):
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((5, -5), polygon) is False

    def test_point_inside_triangle(self):
        polygon = [(0, 0), (10, 0), (5, 10)]
        assert point_in_polygon((5, 3), polygon) is True

    def test_point_outside_triangle(self):
        polygon = [(0, 0), (10, 0), (5, 10)]
        assert point_in_polygon((0, 10), polygon) is False

    def test_invalid_polygon_too_few_vertices(self):
        with pytest.raises(ValueError):
            point_in_polygon((5, 5), [(0, 0), (10, 0)])

    def test_point_inside_complex_polygon(self):
        polygon = [(0, 0), (5, 0), (10, 5), (5, 10), (0, 5)]
        assert point_in_polygon((5, 5), polygon) is True

    def test_point_far_outside(self):
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((100, 100), polygon) is False

    def test_negative_coordinates(self):
        polygon = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
        assert point_in_polygon((0, 0), polygon) is True
        assert point_in_polygon((-20, 0), polygon) is False


# --- Bounding Box Operations ---

class TestBBoxOperations:
    """Tests for bounding box utility functions."""

    def test_bbox_center(self):
        cx, cy = bbox_center([100, 200, 300, 400])
        assert cx == 200.0
        assert cy == 300.0

    def test_bbox_center_small(self):
        cx, cy = bbox_center([0, 0, 10, 10])
        assert cx == 5.0
        assert cy == 5.0

    def test_bbox_center_invalid(self):
        with pytest.raises(ValueError):
            bbox_center([1, 2])

    def test_bbox_area(self):
        assert bbox_area([0, 0, 10, 10]) == 100.0

    def test_bbox_area_zero(self):
        assert bbox_area([5, 5, 5, 5]) == 0.0

    def test_bbox_area_invalid(self):
        assert bbox_area([1]) == 0.0

    def test_bbox_height(self):
        assert bbox_height([0, 0, 10, 20]) == 20.0

    def test_bbox_width(self):
        assert bbox_width([0, 0, 15, 10]) == 15.0

    def test_bbox_bottom_center(self):
        cx, cy = bbox_bottom_center([100, 200, 300, 400])
        assert cx == 200.0
        assert cy == 400.0

    def test_bbox_bottom_center_invalid(self):
        with pytest.raises(ValueError):
            bbox_bottom_center([1])


# --- Coordinate Normalization ---

class TestNormalization:
    """Tests for coordinate normalization."""

    def test_normalize_point(self):
        nx, ny = normalize_point((960, 540), 1920, 1080)
        assert abs(nx - 0.5) < 1e-6
        assert abs(ny - 0.5) < 1e-6

    def test_normalize_origin(self):
        nx, ny = normalize_point((0, 0), 1920, 1080)
        assert nx == 0.0
        assert ny == 0.0

    def test_normalize_invalid_frame(self):
        nx, ny = normalize_point((10, 10), 0, 0)
        assert nx == 0.0
        assert ny == 0.0

    def test_denormalize_polygon(self):
        norm = [(0.5, 0.5), (1.0, 1.0), (0.0, 1.0)]
        result = denormalize_polygon(norm, 1920, 1080)
        assert result[0] == (960, 540)
        assert result[1] == (1920, 1080)
        assert result[2] == (0, 1080)


# --- Velocity Computation ---

class TestVelocity:
    """Tests for velocity computation."""

    def test_velocity_no_movement(self):
        positions = [(5, 5), (5, 5), (5, 5)]
        vx, vy = compute_velocity(positions)
        assert vx == 0.0
        assert vy == 0.0

    def test_velocity_rightward(self):
        positions = [(0, 0), (10, 0), (20, 0)]
        vx, vy = compute_velocity(positions)
        assert vx == 10.0
        assert vy == 0.0

    def test_velocity_downward(self):
        positions = [(0, 0), (0, 5), (0, 10)]
        vx, vy = compute_velocity(positions)
        assert vx == 0.0
        assert vy == 5.0

    def test_velocity_insufficient_data(self):
        vx, vy = compute_velocity([(1, 1)])
        assert vx == 0.0
        assert vy == 0.0

    def test_velocity_empty(self):
        vx, vy = compute_velocity([])
        assert vx == 0.0
        assert vy == 0.0

    def test_velocity_with_timestamps(self):
        positions = [(0, 0), (10, 0)]
        timestamps = [0.0, 2.0]
        vx, vy = compute_velocity(positions, timestamps)
        assert vx == 5.0  # 10 px / 2 sec
        assert vy == 0.0


# --- Distance to Polygon Edge ---

class TestDistanceToPolygon:
    """Tests for distance to polygon edge."""

    def test_distance_at_center_of_square(self):
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        dist = distance_to_polygon_edge((5, 5), polygon)
        assert abs(dist - 5.0) < 0.1

    def test_distance_on_edge(self):
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        dist = distance_to_polygon_edge((5, 0), polygon)
        assert dist < 0.01

    def test_distance_outside(self):
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        dist = distance_to_polygon_edge((15, 5), polygon)
        assert abs(dist - 5.0) < 0.1

    def test_distance_empty_polygon(self):
        dist = distance_to_polygon_edge((5, 5), [(0, 0)])
        assert dist == float('inf')
