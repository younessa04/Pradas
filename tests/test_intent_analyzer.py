"""
Tests for PRADAS Pedestrian Intent Analyzer.

Validates motion classification, danger zone status, orientation inference,
and temporal state smoothing.
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import DangerZoneConfig
from src.tracker import TrackedPedestrian
from src.intent_analyzer import (
    PedestrianIntentAnalyzer,
    IntentState,
    IntentResult
)


@pytest.fixture
def analyzer():
    # Danger zone polygon covering middle area in 640x480
    config = DangerZoneConfig(
        enabled=True,
        polygon_normalized=[(0.25, 0.5), (0.75, 0.5), (0.85, 1.0), (0.15, 1.0)],
        approach_margin_px=50
    )
    return PedestrianIntentAnalyzer(
        danger_zone_config=config,
        frame_width=640,
        frame_height=480
    )


class TestIntentAnalyzer:
    """Tests for PedestrianIntentAnalyzer."""

    def test_unknown_state_insufficient_history(self, analyzer):
        ped = TrackedPedestrian(
            track_id=1,
            bbox=[100, 100, 150, 200],
            confidence=0.9
        )
        # Only 1 observation in history
        ped.history.centers.append(ped.center)
        results = analyzer.analyze([ped])
        assert 1 in results
        assert results[1].state == IntentState.UNKNOWN

    def test_stationary_pedestrian(self, analyzer):
        ped = TrackedPedestrian(
            track_id=2,
            bbox=[50, 50, 100, 150],
            confidence=0.9
        )
        # Pedestrian stays in the same position for 6 frames
        for f in range(6):
            ped.history.centers.append((75.0, 100.0))
            ped.history.timestamps.append(f * 0.1)

        results = analyzer.analyze([ped])
        assert results[2].state == IntentState.STATIONARY
        assert results[2].speed < 2.0

    def test_moving_pedestrian(self, analyzer):
        ped = TrackedPedestrian(
            track_id=3,
            bbox=[50, 50, 100, 150],
            confidence=0.9
        )
        # Pedestrian moving rapidly horizontally
        for f in range(6):
            ped.history.centers.append((50.0 + f * 10.0, 100.0))
            ped.history.timestamps.append(f * 0.1)

        results = analyzer.analyze([ped])
        # Speed should be ~100 px/sec or 10 px/frame
        assert results[3].speed > 5.0
        assert results[3].state in (
            IntentState.MOVING,
            IntentState.POSSIBLE_CROSSING,
            IntentState.ENTERING_DANGER_ZONE
        )

    def test_inside_danger_zone(self, analyzer):
        # Center in danger zone: x=320, y=350 (danger zone is y: 240..480, x: ~160..480)
        ped = TrackedPedestrian(
            track_id=4,
            bbox=[290, 300, 350, 400],
            confidence=0.95
        )
        for f in range(6):
            ped.history.centers.append((320.0, 350.0))
            ped.history.timestamps.append(f * 0.1)

        results = analyzer.analyze([ped])
        assert results[4].in_danger_zone is True
        assert results[4].distance_to_zone_px >= 0.0

    def test_approaching_danger_zone(self, analyzer):
        # Danger zone starts around x=160 at y=360. Put pedestrian at x=140
        ped = TrackedPedestrian(
            track_id=5,
            bbox=[110, 300, 150, 420],
            confidence=0.9
        )
        # Moving toward danger zone
        for f in range(6):
            ped.history.centers.append((100.0 + f * 7.0, 360.0))
            ped.history.timestamps.append(f * 0.1)

        results = analyzer.analyze([ped])
        assert results[5].approaching_danger_zone is True or results[5].in_danger_zone is True

    def test_intent_result_serialization(self):
        result = IntentResult(
            track_id=1,
            state=IntentState.MOVING,
            confidence=0.85,
            velocity_x=4.5,
            velocity_y=0.2,
            speed=4.5,
            direction_degrees=2.5,
            in_danger_zone=False,
            approaching_danger_zone=True,
            distance_to_zone_px=22.4,
            torso_orientation="facing_right"
        )
        d = result.to_dict()
        assert d["track_id"] == 1
        assert d["state"] == "MOVING"
        assert d["confidence"] == 0.85
        assert d["torso_orientation"] == "facing_right"
        assert d["approaching_danger_zone"] is True
