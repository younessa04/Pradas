"""
PRADAS Geometry Utilities.

Point-in-polygon tests, bounding box operations, and coordinate helpers.
"""

import numpy as np
from typing import List, Tuple, Optional


def point_in_polygon(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]]
) -> bool:
    """
    Test if a point lies inside a polygon using ray-casting algorithm.

    Args:
        point: (x, y) coordinates of the test point.
        polygon: List of (x, y) vertices defining the polygon.

    Returns:
        True if the point is inside the polygon.

    Raises:
        ValueError: If polygon has fewer than 3 vertices.
    """
    if len(polygon) < 3:
        raise ValueError("Polygon must have at least 3 vertices.")

    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Compute center of a bounding box [x1, y1, x2, y2].

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        (center_x, center_y) tuple.
    """
    if len(bbox) < 4:
        raise ValueError("Bounding box must have 4 coordinates [x1, y1, x2, y2].")
    x1, y1, x2, y2 = bbox[:4]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: List[float]) -> float:
    """
    Compute area of a bounding box [x1, y1, x2, y2].

    Args:
        bbox: Bounding box coordinates.

    Returns:
        Area in pixels squared.
    """
    if len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = bbox[:4]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def bbox_height(bbox: List[float]) -> float:
    """Get bounding box height."""
    if len(bbox) < 4:
        return 0.0
    return max(0.0, bbox[3] - bbox[1])


def bbox_width(bbox: List[float]) -> float:
    """Get bounding box width."""
    if len(bbox) < 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0])


def bbox_bottom_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Get bottom center of bbox (useful for ground-plane approximation).

    Args:
        bbox: Bounding box [x1, y1, x2, y2].

    Returns:
        (center_x, y2) tuple.
    """
    if len(bbox) < 4:
        raise ValueError("Bounding box must have 4 coordinates.")
    x1, y1, x2, y2 = bbox[:4]
    return ((x1 + x2) / 2.0, y2)


def normalize_point(
    point: Tuple[float, float],
    frame_width: int,
    frame_height: int
) -> Tuple[float, float]:
    """
    Normalize pixel coordinates to [0, 1] range.

    Args:
        point: (x, y) in pixel coordinates.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Normalized (x, y) in [0, 1].
    """
    if frame_width <= 0 or frame_height <= 0:
        return (0.0, 0.0)
    return (point[0] / frame_width, point[1] / frame_height)


def denormalize_polygon(
    polygon_normalized: List[Tuple[float, float]],
    frame_width: int,
    frame_height: int
) -> List[Tuple[int, int]]:
    """
    Convert normalized polygon coordinates to pixel coordinates.

    Args:
        polygon_normalized: List of (x, y) in [0, 1].
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        List of (x, y) in pixel coordinates.
    """
    return [
        (int(x * frame_width), int(y * frame_height))
        for x, y in polygon_normalized
    ]


def compute_velocity(
    positions: List[Tuple[float, float]],
    timestamps: Optional[List[float]] = None,
    window: int = 3
) -> Tuple[float, float]:
    """
    Estimate image-plane velocity from recent positions.

    Uses the last `window` positions to estimate displacement per step.
    If timestamps are provided, velocity is in pixels/second.
    Otherwise, velocity is in pixels/frame.

    Args:
        positions: List of (x, y) positions (oldest first).
        timestamps: Optional list of timestamps corresponding to positions.
        window: Number of recent positions to use.

    Returns:
        (vx, vy) velocity estimate. (0, 0) if insufficient data.
    """
    if len(positions) < 2:
        return (0.0, 0.0)

    recent = positions[-window:]
    if len(recent) < 2:
        return (0.0, 0.0)

    dx = recent[-1][0] - recent[0][0]
    dy = recent[-1][1] - recent[0][1]
    n_steps = len(recent) - 1

    if timestamps and len(timestamps) >= len(recent):
        recent_ts = timestamps[-window:]
        dt = recent_ts[-1] - recent_ts[0]
        if dt > 1e-6:
            return (dx / dt, dy / dt)

    return (dx / n_steps, dy / n_steps)


def distance_to_polygon_edge(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]]
) -> float:
    """
    Approximate minimum distance from point to polygon boundary.

    Args:
        point: (x, y) test point.
        polygon: List of polygon vertices.

    Returns:
        Approximate minimum distance in pixels.
    """
    if len(polygon) < 2:
        return float('inf')

    px, py = point
    min_dist = float('inf')
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Project point onto line segment
        edge_dx = x2 - x1
        edge_dy = y2 - y1
        edge_len_sq = edge_dx * edge_dx + edge_dy * edge_dy

        if edge_len_sq < 1e-10:
            dist = np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        else:
            t = max(0.0, min(1.0,
                ((px - x1) * edge_dx + (py - y1) * edge_dy) / edge_len_sq))
            proj_x = x1 + t * edge_dx
            proj_y = y1 + t * edge_dy
            dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

        min_dist = min(min_dist, dist)

    return min_dist
