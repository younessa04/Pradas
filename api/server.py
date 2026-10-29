"""
PRADAS FastAPI Server.

Provides REST and WebSocket endpoints for the risk assessment system.

IMPORTANT: The API server does NOT instantiate its own copy of AI models.
Model inference happens in the pipeline; the API receives events via
a shared state or event publisher.
"""

import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import StatusResponse, HealthResponse, RiskEventSchema
from api.websocket_manager import WebSocketManager
from utils.logging_config import get_logger

logger = get_logger("api_server")

# Shared state — set by the pipeline when initialized
_app_state = {
    "models_initialized": False,
    "pipeline_status": "idle",
    "last_timestamp": None,
    "device": "cpu",
    "total_frames": 0,
}

# WebSocket manager instance
ws_manager = WebSocketManager(max_queue_size=100)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(
        title="PRADAS API",
        description="Advanced Pedestrian Risk-Aware Driver Assistance System API",
        version="1.0.0",
    )

    # CORS for Flutter/web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/status", response_model=StatusResponse)
    async def get_status() -> StatusResponse:
        """
        Get system status.

        Reports API health, model readiness, and pipeline status separately.
        The API being reachable does NOT mean models are loaded or the
        pipeline is processing.
        """
        return StatusResponse(
            status="running",
            api_version="1.0.0",
            models_initialized=_app_state["models_initialized"],
            active_ws_clients=ws_manager.client_count,
            pipeline_status=_app_state["pipeline_status"],
            last_processing_timestamp=_app_state["last_timestamp"],
            device=_app_state["device"],
            total_frames_processed=_app_state["total_frames"],
        )

    @app.get("/health", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        """
        Health check endpoint.

        Separates API health from model/pipeline readiness.
        """
        return HealthResponse(
            healthy=True,
            api_up=True,
            models_ready=_app_state["models_initialized"],
            pipeline_running=_app_state["pipeline_status"] == "running",
        )

    @app.websocket("/ws/alerts")
    async def websocket_alerts(websocket: WebSocket) -> None:
        """
        WebSocket endpoint for real-time risk alerts.

        Clients connect here to receive streaming pedestrian risk events.
        Messages follow the RiskEventSchema format.
        """
        client_id = await ws_manager.connect(websocket)
        try:
            while True:
                # Keep connection alive; wait for client messages (pings, etc.)
                data = await websocket.receive_text()
                # Echo ping-pong for keep-alive
                if data.strip().lower() == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            await ws_manager.disconnect(client_id)
        except Exception as e:
            logger.warning(f"WebSocket error for {client_id}: {e}")
            await ws_manager.disconnect(client_id)

    return app


def update_app_state(
    models_initialized: bool = None,
    pipeline_status: str = None,
    device: str = None,
    total_frames: int = None
) -> None:
    """
    Update shared application state from the pipeline.

    Args:
        models_initialized: Whether AI models are loaded.
        pipeline_status: Current pipeline status string.
        device: Inference device.
        total_frames: Total frames processed.
    """
    if models_initialized is not None:
        _app_state["models_initialized"] = models_initialized
    if pipeline_status is not None:
        _app_state["pipeline_status"] = pipeline_status
    if device is not None:
        _app_state["device"] = device
    if total_frames is not None:
        _app_state["total_frames"] = total_frames
    _app_state["last_timestamp"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
