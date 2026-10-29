"""
PRADAS API Schemas.

Pydantic models for request/response validation and WebSocket messages.
All models are JSON-serializable with schema versioning.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RiskLevelSchema(str, Enum):
    """Risk level enum for API responses."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class IntentStateSchema(str, Enum):
    """Intent state enum for API responses."""
    UNKNOWN = "UNKNOWN"
    STATIONARY = "STATIONARY"
    MOVING = "MOVING"
    POSSIBLE_CROSSING = "POSSIBLE_CROSSING"
    ENTERING_DANGER_ZONE = "ENTERING_DANGER_ZONE"


class DangerZoneStatusSchema(str, Enum):
    """Danger zone status enum."""
    OUTSIDE = "OUTSIDE"
    APPROACHING = "APPROACHING"
    INSIDE = "INSIDE"
    LEAVING = "LEAVING"
    UNKNOWN = "UNKNOWN"


class PedestrianRiskSchema(BaseModel):
    """Schema for a single pedestrian risk assessment."""
    track_id: int
    risk_level: RiskLevelSchema
    risk_score: float = Field(ge=0.0, le=1.0)
    ttc_seconds: Optional[float] = None
    ttc_method: str = "unavailable"
    ttc_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    relative_depth: Optional[float] = None
    distance_meters: Optional[float] = None
    intent_state: IntentStateSchema = IntentStateSchema.UNKNOWN
    danger_zone_status: DangerZoneStatusSchema = DangerZoneStatusSchema.UNKNOWN
    reason_codes: List[str] = Field(default_factory=list)


class RiskEventSchema(BaseModel):
    """
    WebSocket risk event message schema.

    Compatible with Flutter JSON parsing.
    """
    schema_version: str = "1.0"
    event_type: str = "pedestrian_risk_update"
    timestamp: str  # ISO 8601 UTC
    frame_index: int
    pedestrians: List[PedestrianRiskSchema] = Field(default_factory=list)


class StatusResponse(BaseModel):
    """Response schema for GET /status endpoint."""
    status: str = "running"
    api_version: str = "1.0.0"
    models_initialized: bool = False
    active_ws_clients: int = 0
    pipeline_status: str = "idle"
    last_processing_timestamp: Optional[str] = None
    device: str = "cpu"
    total_frames_processed: int = 0


class HealthResponse(BaseModel):
    """Response schema for GET /health endpoint."""
    healthy: bool = True
    api_up: bool = True
    models_ready: bool = False
    pipeline_running: bool = False
