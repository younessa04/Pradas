"""
Tests for PRADAS API Schemas.
"""

import pytest
import json

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.schemas import (
    PedestrianRiskSchema,
    RiskEventSchema,
    StatusResponse,
    HealthResponse,
    RiskLevelSchema,
    IntentStateSchema,
    DangerZoneStatusSchema,
)


class TestPedestrianRiskSchema:
    """Tests for PedestrianRiskSchema."""

    def test_valid_risk_schema(self):
        schema = PedestrianRiskSchema(
            track_id=1,
            risk_level=RiskLevelSchema.HIGH,
            risk_score=0.85,
            ttc_seconds=2.3,
            ttc_method="bbox_expansion",
            ttc_confidence=0.55,
            relative_depth=0.7,
            distance_meters=None,
            intent_state=IntentStateSchema.POSSIBLE_CROSSING,
            danger_zone_status=DangerZoneStatusSchema.INSIDE,
            reason_codes=["INSIDE_DANGER_ZONE"],
        )
        assert schema.track_id == 1
        assert schema.risk_level == RiskLevelSchema.HIGH

    def test_json_serializable(self):
        schema = PedestrianRiskSchema(
            track_id=1,
            risk_level=RiskLevelSchema.LOW,
            risk_score=0.1,
        )
        json_str = schema.model_dump_json()
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert parsed["track_id"] == 1

    def test_risk_score_bounds(self):
        with pytest.raises(Exception):
            PedestrianRiskSchema(
                track_id=1,
                risk_level=RiskLevelSchema.LOW,
                risk_score=1.5,  # Out of bounds
            )

    def test_null_optional_fields(self):
        schema = PedestrianRiskSchema(
            track_id=1,
            risk_level=RiskLevelSchema.LOW,
            risk_score=0.1,
            ttc_seconds=None,
            distance_meters=None,
        )
        d = schema.model_dump()
        assert d["ttc_seconds"] is None
        assert d["distance_meters"] is None


class TestRiskEventSchema:
    """Tests for WebSocket risk event schema."""

    def test_valid_event(self):
        event = RiskEventSchema(
            schema_version="1.0",
            event_type="pedestrian_risk_update",
            timestamp="2026-10-24T12:00:00Z",
            frame_index=120,
            pedestrians=[
                PedestrianRiskSchema(
                    track_id=1,
                    risk_level=RiskLevelSchema.HIGH,
                    risk_score=0.87,
                    reason_codes=["SHORT_TTC"],
                )
            ]
        )
        assert event.schema_version == "1.0"
        assert len(event.pedestrians) == 1

    def test_empty_pedestrians(self):
        event = RiskEventSchema(
            timestamp="2026-10-24T12:00:00Z",
            frame_index=0,
            pedestrians=[],
        )
        assert len(event.pedestrians) == 0

    def test_event_json_roundtrip(self):
        event = RiskEventSchema(
            timestamp="2026-10-24T12:00:00Z",
            frame_index=100,
            pedestrians=[
                PedestrianRiskSchema(
                    track_id=5,
                    risk_level=RiskLevelSchema.MEDIUM,
                    risk_score=0.55,
                )
            ],
        )
        json_str = event.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["frame_index"] == 100
        assert parsed["pedestrians"][0]["track_id"] == 5


class TestStatusResponse:
    """Tests for status response schema."""

    def test_default_status(self):
        status = StatusResponse()
        assert status.status == "running"
        assert status.models_initialized is False
        assert status.pipeline_status == "idle"

    def test_status_json(self):
        status = StatusResponse(
            models_initialized=True,
            active_ws_clients=3,
            pipeline_status="running",
            device="cuda",
        )
        d = status.model_dump()
        assert d["models_initialized"] is True
        assert d["active_ws_clients"] == 3


class TestHealthResponse:
    """Tests for health response schema."""

    def test_default_health(self):
        health = HealthResponse()
        assert health.healthy is True
        assert health.api_up is True
        assert health.models_ready is False

    def test_health_with_models(self):
        health = HealthResponse(
            models_ready=True,
            pipeline_running=True,
        )
        assert health.models_ready is True
