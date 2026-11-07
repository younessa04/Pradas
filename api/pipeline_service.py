"""
PRADAS — Background Pipeline Service for Real-Time Streaming & WebSocket Telemetry.

Manages running the PedestrianRiskPipeline in a background thread, captures
annotated frames for MJPEG streaming, and broadcasts live vehicle telemetry
(speed, pedestrians, TTC, radar, risk) to connected WebSocket dashboard clients.
"""

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np

from config.settings import Settings, get_settings
from src.pipeline import PedestrianRiskPipeline
from utils.logging_config import get_logger

logger = get_logger("pipeline_service")


class PipelineService:
    """Singleton service to control live video pipeline execution and streaming."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pipeline: Optional[PedestrianRiskPipeline] = None

        self._is_running = False
        self._current_source = "0"
        self._current_model = "models/best_v2.pt"
        self._fps = 0.0
        self._frame_index = 0
        self._total_frames = 0
        self._highest_risk = "NONE"
        self._lowest_ttc: Optional[float] = None
        self._simulated_speed_kmh = 48.0

        self._latest_jpeg: Optional[bytes] = None
        self._latest_telemetry: Dict[str, Any] = {}
        self._ws_manager = None
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None

        # Pre-generate standby image
        self._standby_jpeg = self._create_standby_frame()

    def set_ws_manager(self, ws_manager) -> None:
        """Set WebSocket manager for telemetry broadcast."""
        self._ws_manager = ws_manager

    def set_asyncio_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set active asyncio event loop."""
        self._asyncio_loop = loop

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_status(self) -> Dict[str, Any]:
        """Return current status dictionary."""
        with self._lock:
            return {
                "running": self._is_running,
                "source": self._current_source,
                "model": self._current_model,
                "fps": round(self._fps, 1),
                "frame_index": self._frame_index,
                "total_frames": self._total_frames,
                "highest_risk": self._highest_risk,
                "lowest_ttc": round(self._lowest_ttc, 2) if self._lowest_ttc else None,
                "speed_kmh": round(self._simulated_speed_kmh, 1),
            }

    def list_sources(self) -> List[Dict[str, str]]:
        """List available video input sources (webcams and input files)."""
        sources = [
            {"id": "0", "name": "📷 Caméra WebCam PC (Index 0)", "type": "webcam"}
        ]

        # Scan data/input
        root = Path(__file__).resolve().parent.parent
        input_dir = root / "data" / "input"
        if input_dir.exists():
            for f in sorted(input_dir.glob("*.mp4")):
                sources.append({
                    "id": str(f.relative_to(root)).replace("\\", "/"),
                    "name": f"🎬 {f.stem} (Fichier Vidéo)",
                    "type": "video"
                })

        return sources

    def list_models(self) -> List[Dict[str, str]]:
        """List available trained YOLO models."""
        root = Path(__file__).resolve().parent.parent
        models_dir = root / "models"
        available = []

        if (models_dir / "best_v2.pt").exists():
            available.append({
                "id": "models/best_v2.pt",
                "name": "YOLOv11s — best_v2.pt (CrowdHuman + MOT17 — Recommandé, 90.1% mAP)",
                "recommended": True
            })

        if (models_dir / "best.pt").exists():
            available.append({
                "id": "models/best.pt",
                "name": "YOLOv8s — best.pt (CrowdHuman seul, 86.8% mAP)",
                "recommended": False
            })

        if not available:
            available.append({
                "id": "yolov8n-pose.pt",
                "name": "YOLOv8n Pose (COCO Baseline)",
                "recommended": False
            })

        return available

    def start(
        self,
        source: str = "0",
        model: str = "models/best_v2.pt",
        confidence: float = 0.25,
        imgsz: int = 960,
        frame_skip: int = 1,
        night_mode: str = "auto",
        loop_video: bool = True
    ) -> bool:
        """Start the pipeline execution in background thread."""
        with self._lock:
            if self._is_running:
                logger.warning("Pipeline is already running.")
                return False

            self._stop_event.clear()
            self._current_source = source
            self._current_model = model

            self._thread = threading.Thread(
                target=self._run_loop,
                args=(source, model, confidence, imgsz, frame_skip, night_mode, loop_video),
                daemon=True,
                name="PRADAS-PipelineThread"
            )
            self._is_running = True
            self._thread.start()
            logger.info(f"Pipeline service started with source='{source}', model='{model}'")
            return True

    def stop(self) -> bool:
        """Stop the running pipeline."""
        with self._lock:
            if not self._is_running:
                return True
            self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        with self._lock:
            self._is_running = False
            self._highest_risk = "NONE"
            self._lowest_ttc = None
            self._latest_jpeg = None
            if self._pipeline:
                self._pipeline.reset()
            logger.info("Pipeline service stopped.")
        return True

    def get_latest_frame_bytes(self) -> bytes:
        """Return latest encoded JPEG frame or standby image."""
        with self._lock:
            if self._is_running and self._latest_jpeg is not None:
                return self._latest_jpeg
            return self._standby_jpeg

    def _run_loop(
        self,
        source: str,
        model: str,
        confidence: float,
        imgsz: int,
        frame_skip: int,
        night_mode: str,
        loop_video: bool
    ) -> None:
        """Core execution loop inside background thread."""
        try:
            if (self._pipeline is None or 
                not getattr(self._pipeline, '_models_initialized', False) or 
                getattr(self, '_cached_model', None) != model):
                settings = get_settings()
                settings.detection.yolo_pose_model = model
                settings.detection.confidence_threshold = confidence
                settings.tracking.track_high_thresh = confidence
                settings.tracking.new_track_thresh = confidence
                settings.detection.imgsz = imgsz
                settings.video.frame_skip = max(1, frame_skip)
                settings.night_enhancement.mode = night_mode

                pipeline = PedestrianRiskPipeline(settings=settings)
                pipeline.initialize_models()
                self._pipeline = pipeline
                self._cached_model = model
            else:
                pipeline = self._pipeline
                pipeline.settings.detection.confidence_threshold = confidence
                pipeline.settings.tracking.track_high_thresh = confidence
                pipeline.settings.tracking.new_track_thresh = confidence
                pipeline.settings.video.frame_skip = max(1, frame_skip)
                pipeline.settings.night_enhancement.mode = night_mode

            # Parse source: index if digit, else path
            video_src = int(source) if source.isdigit() else source
            if isinstance(video_src, int):
                cap = cv2.VideoCapture(video_src, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(video_src)
            else:
                cap = cv2.VideoCapture(video_src)

            if not cap.isOpened():
                logger.error(f"Cannot open video source: {source}")
                with self._lock:
                    self._is_running = False
                return

            frame_idx = 0
            base_speed = 50.0

            # Ensure clean state for new session (no leftover tracks from previous video/webcam)
            pipeline.reset()

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    if loop_video and not source.isdigit():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.info("End of video stream.")
                        break

                frame_idx += 1
                if frame_skip > 1 and (frame_idx % frame_skip) != 0:
                    continue

                # Run PRADAS pipeline processing
                annotated, assessments, _ = pipeline.process_frame(frame)
                fps = pipeline._perf_monitor.fps if hasattr(pipeline, '_perf_monitor') else 0.0

                # Only keep assessments for pedestrians actually seen in the current frame
                seen_track_ids = {
                    p.track_id for p in pipeline._tracker._tracks.values()
                    if p.last_seen_frame == frame_idx
                } if pipeline._tracker else set()
                current_assessments = [a for a in assessments if a.track_id in seen_track_ids]

                # Extract risk metrics
                highest_risk = "NONE"
                lowest_ttc = None

                for a in current_assessments:
                    level = a.risk_level.value if hasattr(a.risk_level, 'value') else str(a.risk_level)
                    if level == "HIGH":
                        highest_risk = "HIGH"
                    elif level == "MEDIUM" and highest_risk != "HIGH":
                        highest_risk = "MEDIUM"
                    elif level == "LOW" and highest_risk == "NONE":
                        highest_risk = "LOW"

                    if a.ttc_seconds is not None and a.ttc_seconds > 0:
                        if lowest_ttc is None or a.ttc_seconds < lowest_ttc:
                            lowest_ttc = a.ttc_seconds

                # Simulated adaptive vehicle speed (AEB reaction)
                if highest_risk == "HIGH" or (lowest_ttc is not None and lowest_ttc < 2.5):
                    # Hard Braking!
                    base_speed = max(0.0, base_speed - 8.0)
                elif highest_risk == "MEDIUM":
                    base_speed = max(25.0, base_speed - 2.0)
                else:
                    # Accelerate back to cruising speed
                    base_speed = min(50.0, base_speed + 1.5)

                # Compress annotated frame to JPEG
                if annotated is not None:
                    # Resize slightly for fast web streaming if ultra large
                    h, w = annotated.shape[:2]
                    stream_frame = annotated
                    if w > 1280:
                        scale = 1280 / w
                        stream_frame = cv2.resize(annotated, (1280, int(h * scale)))

                    _, buf = cv2.imencode(".jpg", stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                    jpeg_bytes = buf.tobytes()

                    with self._lock:
                        self._latest_jpeg = jpeg_bytes
                        self._fps = fps
                        self._frame_index = frame_idx
                        self._total_frames = frame_idx
                        self._highest_risk = highest_risk
                        self._lowest_ttc = lowest_ttc
                        self._simulated_speed_kmh = base_speed

                # Format detailed telemetry message
                telemetry = {
                    "event_type": "telemetry_update",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "frame_index": frame_idx,
                    "fps": round(fps, 1),
                    "speed_kmh": round(base_speed, 1),
                    "active_pedestrians_count": len(current_assessments),
                    "highest_risk": highest_risk,
                    "lowest_ttc": round(lowest_ttc, 2) if lowest_ttc is not None else None,
                    "collision_warning": highest_risk == "HIGH" or (lowest_ttc is not None and lowest_ttc < 2.5),
                    "pedestrians": [
                        {
                            "track_id": int(a.track_id),
                            "distance_m": round(float(a.distance_meters), 1) if a.distance_meters is not None else None,
                            "ttc_s": round(float(a.ttc_seconds), 1) if a.ttc_seconds is not None else None,
                            "risk_level": a.risk_level.value if hasattr(a.risk_level, 'value') else str(a.risk_level),
                            "risk_score": round(float(a.risk_score), 2),
                            "intent": a.intent_state.value if hasattr(a.intent_state, 'value') else str(a.intent_state),
                            "danger_zone": a.danger_zone_status.value if hasattr(a.danger_zone_status, 'value') else str(a.danger_zone_status),
                        }
                        for a in current_assessments
                    ]
                }

                # Broadcast via WebSocket
                if self._ws_manager and self._asyncio_loop and self._ws_manager.has_clients:
                    asyncio.run_coroutine_threadsafe(
                        self._ws_manager.broadcast(telemetry),
                        self._asyncio_loop
                    )

            cap.release()
            pipeline.cleanup()

        except Exception as e:
            logger.error(f"Error in pipeline service thread: {e}", exc_info=True)
        finally:
            with self._lock:
                self._is_running = False
                self._highest_risk = "NONE"
                self._lowest_ttc = None

    def _create_standby_frame(self) -> bytes:
        """Create a futuristic automotive HUD standby screen."""
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[:] = (15, 20, 28)  # Deep carbon blue

        # Grid lines
        for y in range(0, 720, 60):
            cv2.line(img, (0, y), (1280, y), (25, 32, 45), 1)
        for x in range(0, 1280, 80):
            cv2.line(img, (x, 0), (x, 720), (25, 32, 45), 1)

        # Center target box
        cv2.rectangle(img, (340, 200), (940, 520), (0, 240, 255), 2)
        cv2.line(img, (320, 360), (360, 360), (0, 240, 255), 3)
        cv2.line(img, (920, 360), (960, 360), (0, 240, 255), 3)
        cv2.line(img, (640, 180), (640, 220), (0, 240, 255), 3)
        cv2.line(img, (640, 500), (640, 540), (0, 240, 255), 3)

        cv2.putText(
            img, "PRADAS — ADAS VISION SYSTEM",
            (400, 300), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2
        )
        cv2.putText(
            img, "SYSTEM STANDBY — EN ATTENTE D'ACTIVATION",
            (380, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 240, 255), 2
        )
        cv2.putText(
            img, "Selectionnez une source et cliquez sur 'DEMARRER LE SYSTEME'",
            (360, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 180, 200), 1
        )

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()


# Global singleton instance
pipeline_service = PipelineService()

