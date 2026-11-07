"""
PRADAS FastAPI Server & Automotive Digital Cockpit Backend.

Provides:
- Modern Tesla-style Automotive Cockpit Dashboard UI on GET /
- Real-time MJPEG Video Streaming on GET /video_feed
- WebSocket Telemetry & Forward Collision Warning (FCW) on /ws/alerts
- REST endpoints for dynamic video source switching (Webcam / MOT17 files)
- Status, Health, and Model selection APIs
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from api.schemas import StatusResponse, HealthResponse, RiskEventSchema
from api.websocket_manager import WebSocketManager
from api.pipeline_service import pipeline_service
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


class PipelineStartRequest(BaseModel):
    """Request payload to start real-time ADAS pipeline."""
    source: str = Field(default="0", description="Camera index (e.g. '0') or video file path")
    model: str = Field(default="models/best_v2.pt", description="Path to trained YOLO weights")
    confidence: float = Field(default=0.25, ge=0.05, le=0.95)
    imgsz: int = Field(default=960, ge=320, le=1920)
    frame_skip: int = Field(default=1, ge=1, le=10)
    night_mode: str = Field(default="auto")
    loop_video: bool = Field(default=True)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="PRADAS API & Cockpit",
        description="Pedestrian Risk-Aware Driver Assistance System — Automotive Digital Cockpit",
        version="2.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event() -> None:
        loop = asyncio.get_running_loop()
        pipeline_service.set_asyncio_loop(loop)
        pipeline_service.set_ws_manager(ws_manager)
        logger.info("PRADAS API & Cockpit initialized.")

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        pipeline_service.stop()

    # ─── 1. Cockpit Web UI ──────────────────────────────────────────────────
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return HTMLResponse(content="", status_code=204)

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard() -> HTMLResponse:
        """Serve the modern automotive digital cockpit dashboard."""
        html_path = Path(__file__).resolve().parent / "templates" / "dashboard.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>PRADAS Cockpit: template not found</h1>", status_code=404)

    # ─── 2. Video Feed Streaming (MJPEG) ───────────────────────────────────
    @app.get("/video_feed")
    async def video_feed() -> StreamingResponse:
        """Stream real-time processed ADAS video to browser."""
        async def frame_stream():
            while True:
                frame_bytes = pipeline_service.get_latest_frame_bytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
                await asyncio.sleep(0.04)  # ~25 FPS

        return StreamingResponse(
            frame_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    # ─── 3. Source & Model Management Endpoints ────────────────────────────
    @app.get("/api/sources")
    async def get_sources():
        """List available cameras and input videos."""
        return pipeline_service.list_sources()

    @app.get("/api/models")
    async def get_models():
        """List available trained YOLO pedestrian models."""
        return pipeline_service.list_models()

    @app.post("/api/pipeline/start")
    async def start_pipeline(req: PipelineStartRequest):
        """Start real-time detection on selected source."""
        try:
            loop = asyncio.get_running_loop()
            pipeline_service.set_asyncio_loop(loop)
        except RuntimeError:
            pass
        pipeline_service.set_ws_manager(ws_manager)

        started = pipeline_service.start(
            source=req.source,
            model=req.model,
            confidence=req.confidence,
            imgsz=req.imgsz,
            frame_skip=req.frame_skip,
            night_mode=req.night_mode,
            loop_video=req.loop_video
        )
        if started:
            update_app_state(models_initialized=True, pipeline_status="running")
        return {"success": started, "status": pipeline_service.get_status()}

    @app.post("/api/pipeline/stop")
    async def stop_pipeline():
        """Stop real-time detection."""
        stopped = pipeline_service.stop()
        update_app_state(pipeline_status="idle")
        return {"success": stopped, "status": pipeline_service.get_status()}

    @app.get("/api/pipeline/status")
    async def get_pipeline_status():
        """Get live pipeline status and metrics."""
        return pipeline_service.get_status()

    # ─── 4. Health & Status Endpoints (Preserved for tests) ────────────────
    @app.get("/status", response_model=StatusResponse)
    async def get_status() -> StatusResponse:
        """Get system status."""
        svc_status = pipeline_service.get_status()
        is_running = svc_status["running"]
        return StatusResponse(
            status="running",
            api_version="1.0.0",
            models_initialized=_app_state["models_initialized"] or is_running,
            active_ws_clients=ws_manager.client_count,
            pipeline_status="running" if is_running else _app_state["pipeline_status"],
            last_processing_timestamp=_app_state["last_timestamp"],
            device=_app_state["device"],
            total_frames_processed=svc_status["total_frames"] if is_running else _app_state["total_frames"],
        )

    @app.get("/health", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        """Health check endpoint."""
        svc_status = pipeline_service.get_status()
        is_running = svc_status["running"]
        return HealthResponse(
            healthy=True,
            api_up=True,
            models_ready=_app_state["models_initialized"] or is_running,
            pipeline_running=is_running or (_app_state["pipeline_status"] == "running"),
        )

    # ─── 5. WebSocket Telemetry & Alerts ──────────────────────────────────
    @app.websocket("/ws/alerts")
    async def websocket_alerts(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time risk alerts and vehicle telemetry."""
        loop = asyncio.get_running_loop()
        pipeline_service.set_asyncio_loop(loop)
        pipeline_service.set_ws_manager(ws_manager)

        client_id = await ws_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
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
    """Update shared application state."""
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
