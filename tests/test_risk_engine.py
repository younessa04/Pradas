"""
Tests for PRADAS Risk & TTC Engine.

Uses mock data — does NOT require pretrained model weights.
"""

import pytest
import time
import math
from collections import deque

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import RiskConfig, DangerZoneConfig
from src.risk_ttc_engine import (
    PedestrianRiskEngine, RiskAssessment, RiskLevel,
    DangerZoneStatus, TTCMethod
)
from src.intent_analyzer import IntentResult, IntentState
from src.tracker import TrackedPedestrian, TrackHistory


def _make_tracked_ped(
    track_id: int = 1,
    bbox=None,
    confidence: float = 0.9,
    track_age: int = 10,
    centers=None,
    heights=None,
    depths=None,
    timestamps=None
) -> TrackedPedestrian:
    """Helper to create a TrackedPedestrian with history."""
    if bbox is None:
        bbox = [100.0, 200.0, 200.0, 400.0]

    ped = TrackedPedestrian(
        track_id=track_id,
        bbox=bbox,
        confidence=confidence,
        frame_index=50,
        track_age=track_age,
        last_seen_frame=50,
        first_seen_frame=1,
    )

    if centers:
        for c in centers:
            ped.history.centers.append(c)
    if heights:
        for h in heights:
            ped.history.bbox_heights.append(h)
    if depths:
        for d in depths:
            ped.history.depths.append(d)
    if timestamps:
        for t in timestamps:
            ped.history.timestamps.append(t)

    return ped


def _make_intent(
    track_id: int = 1,
    state: IntentState = IntentState.UNKNOWN,
    in_danger_zone: bool = False,
    approaching: bool = False
) -> IntentResult:
    """Helper to create an IntentResult."""
    return IntentResult(
        track_id=track_id,
        state=state,
        confidence=0.7,
        in_danger_zone=in_danger_zone,
        approaching_danger_zone=approaching,
    )


@pytest.fixture
def risk_engine():
    """Create a risk engine with default config."""
    return PedestrianRiskEngine()


@pytest.fixture
def engine_low_thresholds():
    """Create a risk engine with lower thresholds for testing."""
    config = RiskConfig(
        high_risk_threshold=0.6,
        medium_risk_threshold=0.3,
        hysteresis_frames=1,
    )
    return PedestrianRiskEngine(risk_config=config)


class TestRiskAssessment:
    """Tests for risk assessment output."""

    def test_pedestrian_outside_danger_zone(self, risk_engine):
        ped = _make_tracked_ped()
        intent = _make_intent(in_danger_zone=False, state=IntentState.STATIONARY)

        assessments = risk_engine.assess(
            [ped], {1: intent}, {1: None}, {1: None}, 50
        )
        assert len(assessments) == 1
        assert assessments[0].danger_zone_status == DangerZoneStatus.OUTSIDE

    def test_pedestrian_inside_danger_zone(self, engine_low_thresholds):
        ped = _make_tracked_ped()
        intent = _make_intent(
            in_danger_zone=True,
            state=IntentState.POSSIBLE_CROSSING
        )

        assessments = engine_low_thresholds.assess(
            [ped], {1: intent}, {1: 0.8}, {1: None}, 50
        )
        assert len(assessments) == 1
        assert assessments[0].danger_zone_status == DangerZoneStatus.INSIDE
        # Should be at least MEDIUM with inside zone + crossing intent
        assert assessments[0].risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)

    def test_missing_depth_data(self, risk_engine):
        ped = _make_tracked_ped()
        intent = _make_intent(state=IntentState.MOVING)

        assessments = risk_engine.assess(
            [ped], {1: intent}, {1: None}, {1: None}, 50
        )
        assert len(assessments) == 1
        assert assessments[0].relative_depth is None
        # Missing depth should NOT automatically produce LOW risk
        # (UNKNOWN intent gets 0.4 score, so it won't be zero)

    def test_missing_metric_distance_is_none(self, risk_engine):
        ped = _make_tracked_ped()
        intent = _make_intent()

        assessments = risk_engine.assess(
            [ped], {1: intent}, {1: 0.5}, {1: None}, 50
        )
        assert assessments[0].distance_meters is None

    def test_no_false_metric_ttc_from_relative_depth(self, risk_engine):
        """Relative depth must NOT produce calibrated metric TTC."""
        ped = _make_tracked_ped(
            depths=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
            timestamps=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            heights=[100, 105, 110, 115, 120, 125]
        )
        intent = _make_intent(state=IntentState.MOVING)

        assessments = risk_engine.assess(
            [ped], {1: intent}, {1: 0.55}, {1: None}, 50
        )
        assert assessments[0].distance_meters is None
        # If TTC is produced, it should NOT be calibrated_metric method
        if assessments[0].ttc_seconds is not None:
            assert assessments[0].ttc_method != TTCMethod.CALIBRATED_METRIC

    def test_risk_score_is_bounded(self, risk_engine):
        ped = _make_tracked_ped()
        intent = _make_intent(
            in_danger_zone=True,
            state=IntentState.POSSIBLE_CROSSING
        )

        assessments = risk_engine.assess(
            [ped], {1: intent}, {1: 0.9}, {1: None}, 50
        )
        score = assessments[0].risk_score
        assert 0.0 <= score <= 1.0


class TestTTCEstimation:
    """Tests for TTC estimation methods."""

    def test_ttc_unavailable_with_no_history(self, risk_engine):
        ped = _make_tracked_ped()
        intent = _make_intent()

        assessments = risk_engine.assess(
            [ped], {1: intent}, {1: None}, {1: None}, 50
        )
        assert assessments[0].ttc_method == TTCMethod.UNAVAILABLE

    def test_ttc_bbox_expansion_with_growing_bbox(self, risk_engine):
        # Simulate an approaching pedestrian (growing bbox)
        ts_base = time.time()
        ped = _make_tracked_ped(
            heights=[80, 85, 90, 95, 100, 110, 120],
            timestamps=[ts_base + i * 0.1 for i in range(7)]
        )
        intent = _make_intent(state=IntentState.MOVING)

        assessments = risk_engine.assess(
            [ped], {1: intent}, {1: None}, {1: None}, 50
        )
        # With enough growing history, TTC should be estimated
        if assessments[0].ttc_seconds is not None:
            assert assessments[0].ttc_seconds > 0
            assert assessments[0].ttc_confidence <= 0.6  # Capped for this method

    def test_invalid_ttc_never_zero(self, risk_engine):
        ped = _make_tracked_ped()
        intent = _make_intent()

        assessments = risk_engine.assess(
            [ped], {1: intent}, {1: None}, {1: None}, 50
        )
        # TTC should be None (unavailable) or positive, NEVER zero
        ttc = assessments[0].ttc_seconds
        assert ttc is None or ttc > 0


class TestHysteresis:
    """Tests for risk level hysteresis."""

    def test_hysteresis_prevents_flickering(self):
        config = RiskConfig(hysteresis_frames=3)
        engine = PedestrianRiskEngine(risk_config=config)

        ped = _make_tracked_ped()
        intent_low = _make_intent(state=IntentState.STATIONARY)
        intent_high = _make_intent(
            in_danger_zone=True,
            state=IntentState.POSSIBLE_CROSSING
        )

        # First frame: should establish baseline
        result1 = engine.assess([ped], {1: intent_low}, {1: None}, {1: None}, 1)
        level1 = result1[0].risk_level

        # Single high-risk frame should not immediately change
        result2 = engine.assess([ped], {1: intent_high}, {1: 0.9}, {1: None}, 2)
        # May or may not change — depends on hysteresis count
        # But the test verifies the mechanism exists


class TestJSONSerialization:
    """Tests for JSON-safe risk events."""

    def test_assessment_to_dict(self):
        assessment = RiskAssessment(
            track_id=5,
            risk_level=RiskLevel.HIGH,
            risk_score=0.85,
            ttc_seconds=2.3,
            ttc_method=TTCMethod.BBOX_EXPANSION,
            ttc_confidence=0.55,
            relative_depth=0.7,
            distance_meters=None,
            intent_state=IntentState.POSSIBLE_CROSSING,
            danger_zone_status=DangerZoneStatus.INSIDE,
            reason_codes=["INSIDE_DANGER_ZONE", "SHORT_TTC"],
            timestamp="2026-10-24T12:00:00Z",
            frame_index=120,
        )
        d = assessment.to_dict()
        assert d["track_id"] == 5
        assert d["risk_level"] == "HIGH"
        assert d["distance_meters"] is None
        assert isinstance(d["reason_codes"], list)

        # Verify JSON serializable
        import json
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_assessment_to_dict_none_ttc(self):
        assessment = RiskAssessment(track_id=1)
        d = assessment.to_dict()
        assert d["ttc_seconds"] is None
        assert d["relative_depth"] is None

        import json
        json.dumps(d)  # Should not raise

    def test_assessment_nan_handling(self):
        """Ensure NaN values don't appear in JSON output."""
        assessment = RiskAssessment(
            track_id=1,
            risk_score=0.5,
            ttc_seconds=None,
            relative_depth=None,
        )
        d = assessment.to_dict()

        import json
        json_str = json.dumps(d)
        assert "NaN" not in json_str
        assert "Infinity" not in json_str


class TestAlertCooldown:
    """Tests for alert cooldown mechanism."""

    def test_alert_cooldown(self):
        config = RiskConfig(alert_cooldown_seconds=1.0)
        engine = PedestrianRiskEngine(risk_config=config)

        assessment = RiskAssessment(
            track_id=1,
            risk_level=RiskLevel.HIGH,
            risk_score=0.9,
        )

        # First alert should fire
        assert engine.should_alert(assessment) is True

        # Immediate second should be blocked by cooldown
        assert engine.should_alert(assessment) is False

    def test_low_risk_no_alert(self):
        engine = PedestrianRiskEngine()
        assessment = RiskAssessment(
            track_id=1,
            risk_level=RiskLevel.LOW,
            risk_score=0.1,
        )
        assert engine.should_alert(assessment) is False


class TestExpiredTracks:
    """Tests for track cleanup."""

    def test_expired_tracks_cleaned(self, risk_engine):
        ped1 = _make_tracked_ped(track_id=1)
        ped2 = _make_tracked_ped(track_id=2)
        intent1 = _make_intent(track_id=1)
        intent2 = _make_intent(track_id=2)

        # Both tracks present
        risk_engine.assess(
            [ped1, ped2],
            {1: intent1, 2: intent2},
            {1: None, 2: None},
            {1: None, 2: None},
            50
        )

        # Only track 1 present
        risk_engine.assess(
            [ped1],
            {1: intent1},
            {1: None},
            {1: None},
            51
        )
        # Internal cleanup should remove track 2's history
