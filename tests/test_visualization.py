"""
Tests for PRADAS Frame Visualizer.
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.visualization import FrameVisualizer
from src.tracker import TrackedPedestrian
from src.risk_ttc_engine import RiskAssessment, RiskLevel, DangerZoneStatus
from src.intent_analyzer import IntentResult, IntentState


class TestFrameVisualizer:
    """Tests for FrameVisualizer drawing operations."""

    @pytest.fixture
    def visualizer(self):
        return FrameVisualizer(
            show_skeleton=True,
            show_debug=True,
            show_depth_indicator=True,
            show_danger_zone=True
        )

    def test_draw_does_not_mutate_original_frame(self, visualizer):
        original = np.zeros((480, 640, 3), dtype=np.uint8)
        copy_orig = original.copy()

        ped = TrackedPedestrian(
            track_id=1,
            bbox=[100, 100, 200, 300],
            confidence=0.9
        )
        risk = RiskAssessment(
            track_id=1,
            risk_level=RiskLevel.HIGH,
            risk_score=0.85,
            ttc_seconds=1.5,
            danger_zone_status=DangerZoneStatus.INSIDE
        )
        intent = IntentResult(
            track_id=1,
            state=IntentState.ENTERING_DANGER_ZONE,
            confidence=0.8
        )

        annotated = visualizer.draw(
            frame=original,
            tracked_pedestrians=[ped],
            risk_assessments=[risk],
            intent_results={1: intent},
            danger_zone_px=[(100, 200), (500, 200), (600, 450), (50, 450)],
            fps=30.0,
            frame_index=1,
            device="cpu",
            frame_time_ms=33.3
        )

        # Original frame must be unmodified
        np.testing.assert_array_equal(original, copy_orig)
        # Annotated frame must have differences (e.g. drawn bounding box, text, polygon)
        assert not np.array_equal(annotated, original)
        assert annotated.shape == original.shape

    def test_draw_empty_frame(self, visualizer):
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        assert visualizer.draw(empty, [], [], {}) is not None
        assert visualizer.draw(None, [], [], {}) is not None
