"""
PRADAS Central Processing Pipeline.

Orchestrates all modules: enhancement, detection, tracking, depth estimation,
intent analysis, risk assessment, visualization, and event publication.

Models are initialized once. Frame processing follows a defined stage order.
"""

import os
import sys
import time
import cv2
import numpy as np
from typing import Dict, List, Optional

from config.settings import Settings, get_settings
from src.vision_enhancer import VisionEnhancer
from src.detector_pose import PedestrianDetectorPose, PedestrianDetection
from src.tracker import PedestrianTracker, TrackedPedestrian
from src.depth_estimator import MiDaSDepthEstimator
from src.intent_analyzer import PedestrianIntentAnalyzer, IntentResult
from src.risk_ttc_engine import PedestrianRiskEngine, RiskAssessment
from src.visualization import FrameVisualizer
from utils.performance import PerformanceMonitor
from utils.logging_config import setup_logging, get_logger

logger = get_logger("pipeline")


class PedestrianRiskPipeline:
    """
    Central orchestrator for the PRADAS system.

    Manages the full processing pipeline from video input to annotated
    output with risk assessments.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Args:
            settings: System configuration. Uses defaults if None.
        """
        self.settings = settings or get_settings()
        self._device = self.settings.resolve_device()

        # Initialize logging
        setup_logging(level=self.settings.log_level)

        # Modules (initialized lazily)
        self._enhancer: Optional[VisionEnhancer] = None
        self._detector: Optional[PedestrianDetectorPose] = None
        self._tracker: Optional[PedestrianTracker] = None
        self._depth_estimator: Optional[MiDaSDepthEstimator] = None
        self._intent_analyzer: Optional[PedestrianIntentAnalyzer] = None
        self._risk_engine: Optional[PedestrianRiskEngine] = None
        self._visualizer: Optional[FrameVisualizer] = None
        self._perf_monitor: Optional[PerformanceMonitor] = None

        # Video I/O
        self._cap: Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_index: int = 0
        self._frame_width: int = 0
        self._frame_height: int = 0

        # State
        self._models_initialized: bool = False
        self._running: bool = False

        # Event callback for WebSocket publication
        self._event_callback = None

        logger.info(f"Pipeline created: device={self._device}")

    def set_event_callback(self, callback) -> None:
        """
        Set callback for publishing risk events (e.g., to WebSocket).

        Args:
            callback: Callable that accepts a dict (risk event message).
        """
        self._event_callback = callback

    def initialize_models(self) -> bool:
        """
        Initialize all AI models and processing modules.

        Models are loaded ONCE here, not per frame.

        Returns:
            True if all critical models initialized successfully.
        """
        logger.info("=" * 60)
        logger.info("Initializing PRADAS models...")
        logger.info(f"Device: {self._device}")
        logger.info("=" * 60)

        success = True

        # 1. Vision enhancer (always available, lightweight)
        try:
            self._enhancer = VisionEnhancer(self.settings.night_enhancement)
            logger.info("✓ Vision enhancer ready.")
        except Exception as e:
            logger.error(f"✗ Vision enhancer failed: {e}")
            success = False

        # 2. Pedestrian detector + pose
        try:
            self._detector = PedestrianDetectorPose(self.settings.detection)
            if not self._detector.initialize(self._device):
                logger.error("✗ Detector initialization returned False.")
                success = False
            else:
                logger.info("✓ Pedestrian detector ready.")
        except Exception as e:
            logger.error(f"✗ Detector failed: {e}")
            success = False

        # 3. Tracker
        try:
            self._tracker = PedestrianTracker(
                self.settings.tracking,
                device=self._device,
                imgsz=self.settings.detection.imgsz
            )
            if self._detector and self._detector._model:
                self._tracker.set_model(self._detector._model)
            logger.info("✓ Tracker ready.")
        except Exception as e:
            logger.error(f"✗ Tracker failed: {e}")
            success = False

        # 4. Depth estimator
        try:
            self._depth_estimator = MiDaSDepthEstimator(self.settings.depth)
            if not self._depth_estimator.initialize(self._device):
                logger.warning("⚠ Depth estimator failed to load. "
                               "Depth features will be disabled.")
                self._depth_estimator = None
            else:
                logger.info("✓ Depth estimator ready.")
        except Exception as e:
            logger.warning(f"⚠ Depth estimator unavailable: {e}")
            self._depth_estimator = None

        # 5. Risk engine
        try:
            self._risk_engine = PedestrianRiskEngine(
                self.settings.risk, self.settings.danger_zone
            )
            logger.info("✓ Risk engine ready.")
        except Exception as e:
            logger.error(f"✗ Risk engine failed: {e}")
            success = False

        # 6. Visualizer
        try:
            self._visualizer = FrameVisualizer()
            logger.info("✓ Visualizer ready.")
        except Exception as e:
            logger.error(f"✗ Visualizer failed: {e}")
            success = False

        # 7. Performance monitor
        self._perf_monitor = PerformanceMonitor()

        self._models_initialized = success
        logger.info(f"Model initialization {'COMPLETE' if success else 'INCOMPLETE'}.")
        return success

    def open_video_source(self, source: Optional[str] = None) -> bool:
        """
        Open the video source (file, webcam, or RTSP).

        Args:
            source: Video source. If None, uses config value.
                Can be a file path, camera index (as string), or RTSP URL.

        Returns:
            True if the video source was opened successfully.
        """
        src = source or self.settings.video.source

        # Handle directory containing image sequences (such as MOT17 datasets)
        if os.path.isdir(src):
            img_dir = os.path.join(src, "img1") if os.path.isdir(os.path.join(src, "img1")) else src
            sample_file = os.path.join(img_dir, "000001.jpg")
            if os.path.exists(sample_file):
                src = os.path.join(img_dir, "%06d.jpg")
                logger.info(f"Detected MOT sequence directory, resolved to pattern: {src}")

        # Try to parse as camera index
        try:
            cam_idx = int(src)
            self._cap = cv2.VideoCapture(cam_idx)
            logger.info(f"Opening camera index: {cam_idx}")
        except ValueError:
            # File path, pattern (%06d.jpg), or RTSP URL
            if not os.path.exists(src) and not src.startswith("rtsp") and "%" not in src:
                logger.error(f"Video source not found: {src}")
                return False
            self._cap = cv2.VideoCapture(src)
            logger.info(f"Opening video source: {src}")

        if not self._cap.isOpened():
            logger.error(f"Failed to open video source: {src}")
            return False

        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {self._frame_width}x{self._frame_height} @ {fps:.1f} FPS, "
                    f"~{total_frames} frames")

        # Initialize intent analyzer with frame dimensions
        self._intent_analyzer = PedestrianIntentAnalyzer(
            danger_zone_config=self.settings.danger_zone,
            frame_width=self._frame_width,
            frame_height=self._frame_height
        )

        # Open output video writer if configured
        output_path = self.settings.video.output_path
        if output_path:
            self._setup_video_writer(output_path, fps)

        return True

    def _setup_video_writer(self, output_path: str, fps: float) -> None:
        """Set up output video writer with correct real-time playback framerate."""
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.settings.video.output_codec)
            frame_skip = max(1, self.settings.video.frame_skip)
            effective_fps = (self.settings.video.output_fps or fps) / frame_skip

            self._writer = cv2.VideoWriter(
                output_path, fourcc,
                effective_fps,
                (self._frame_width, self._frame_height)
            )
            if self._writer.isOpened():
                logger.info(f"Output video: {output_path} ({effective_fps:.1f} FPS, 1.0x real-time speed)")
            else:
                logger.warning(f"Failed to create video writer: {output_path}")
                self._writer = None
        except Exception as e:
            logger.error(f"Video writer setup error: {e}")
            self._writer = None

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame through the full pipeline.

        Args:
            frame: BGR input frame.

        Returns:
            Tuple of (annotated_frame, risk_assessments, frame_index).
        """
        if frame is None or frame.size == 0:
            return frame, [], self._frame_index

        self._perf_monitor.start_frame()
        timestamp = time.time()

        # Stage 1: Low-light enhancement
        with self._perf_monitor.stage("enhancement"):
            if self._enhancer:
                frame = self._enhancer.enhance(frame)

        # Stage 2: Detection and pose
        with self._perf_monitor.stage("detection"):
            detections = []
            if self._detector and self._detector.is_initialized:
                detections = self._detector.detect(frame, self._frame_index)

        # Stage 3: Tracking
        with self._perf_monitor.stage("tracking"):
            tracked: List[TrackedPedestrian] = []
            if self._tracker:
                tracked = self._tracker.update(
                    frame, detections, self._frame_index, timestamp
                )

        # Stage 4: Depth estimation
        depth_values: Dict[int, Optional[float]] = {}
        metric_distances: Dict[int, Optional[float]] = {}
        depth_map = None

        with self._perf_monitor.stage("depth"):
            if self._depth_estimator and self._depth_estimator.is_initialized:
                depth_map = self._depth_estimator.estimate_depth(frame)
                if depth_map is not None:
                    for ped in tracked:
                        rel_depth = self._depth_estimator.get_pedestrian_depth(
                            depth_map, ped.bbox
                        )
                        depth_values[ped.track_id] = rel_depth
                        # Update track history with depth
                        if rel_depth is not None:
                            ped.history.depths.append(rel_depth)
                        # Metric distance (only if calibrated)
                        metric_dist = self._depth_estimator.get_metric_distance(rel_depth)
                        metric_distances[ped.track_id] = metric_dist

        # Stage 5: Intent analysis
        intent_results: Dict[int, IntentResult] = {}
        with self._perf_monitor.stage("intent"):
            if self._intent_analyzer:
                intent_results = self._intent_analyzer.analyze(tracked)

        # Stage 6: Risk assessment
        risk_assessments: List[RiskAssessment] = []
        with self._perf_monitor.stage("risk"):
            if self._risk_engine:
                risk_assessments = self._risk_engine.assess(
                    tracked, intent_results, depth_values, metric_distances,
                    self._frame_index,
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                )

        # Stage 7: Visualization
        with self._perf_monitor.stage("visualization"):
            danger_zone_px = None
            if self._intent_analyzer:
                danger_zone_px = self._intent_analyzer.danger_zone_pixels

            annotated = frame
            if self._visualizer:
                annotated = self._visualizer.draw(
                    frame,
                    tracked,
                    risk_assessments,
                    intent_results,
                    danger_zone_px=danger_zone_px,
                    fps=self._perf_monitor.fps,
                    frame_index=self._frame_index,
                    device=self._device,
                    frame_time_ms=self._perf_monitor.frame_time_ms,
                )

        # Stage 8: Publish events
        if self._event_callback and risk_assessments:
            alertable = self._risk_engine.get_alertable_assessments(risk_assessments) if self._risk_engine else []
            if alertable:
                event = {
                    "schema_version": "1.0",
                    "event_type": "pedestrian_risk_update",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "frame_index": self._frame_index,
                    "pedestrians": [a.to_dict() for a in alertable]
                }
                try:
                    self._event_callback(event)
                except Exception as e:
                    logger.debug(f"Event callback error: {e}")

        frame_time = self._perf_monitor.end_frame()
        self._perf_monitor.log_summary(interval_frames=100)

        return annotated, risk_assessments, self._frame_index

    def run(
        self,
        source: Optional[str] = None,
        headless: bool = False,
        max_frames: Optional[int] = None
    ) -> None:
        """
        Run the full pipeline loop.

        Args:
            source: Video source override.
            headless: If True, don't show display window.
            max_frames: Maximum frames to process (None = unlimited).
        """
        if not self._models_initialized:
            logger.info("Models not initialized. Initializing now...")
            if not self.initialize_models():
                logger.error("Model initialization failed. Cannot run pipeline.")
                return

        if not self.open_video_source(source):
            logger.error("Failed to open video source. Cannot run pipeline.")
            return

        self._running = True
        self._frame_index = 0
        frame_skip = max(1, self.settings.video.frame_skip)

        logger.info("=" * 60)
        logger.info("PRADAS Pipeline STARTING")
        logger.info(f"  Source: {source or self.settings.video.source}")
        logger.info(f"  Device: {self._device}")
        logger.info(f"  Display: {not headless}")
        logger.info(f"  Frame skip: {frame_skip}")
        logger.info("  Press 'q' to quit (if display is enabled)")
        logger.info("=" * 60)

        try:
            while self._running:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    logger.info("End of video stream.")
                    break

                self._frame_index += 1

                # Frame skipping
                if frame_skip > 1 and (self._frame_index % frame_skip) != 0:
                    continue

                # Max frames limit
                if max_frames and self._frame_index > max_frames:
                    logger.info(f"Reached max_frames limit: {max_frames}")
                    break

                # Process frame
                annotated, assessments, _ = self.process_frame(frame)

                # Write output video
                if self._writer and annotated is not None:
                    self._writer.write(annotated)

                # Display
                if not headless and annotated is not None:
                    cv2.imshow("PRADAS - Pedestrian Risk Assessment", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        logger.info("User requested quit.")
                        break

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user (Ctrl+C).")
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release all resources gracefully."""
        self._running = False
        logger.info("Cleaning up pipeline resources...")

        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Video capture released.")

        if self._writer and self._writer.isOpened():
            self._writer.release()
            logger.info("Video writer released.")

        cv2.destroyAllWindows()

        if self._perf_monitor:
            logger.info("\n" + self._perf_monitor.get_summary())

        logger.info("Pipeline cleanup complete.")

    @property
    def is_running(self) -> bool:
        """Whether the pipeline is currently running."""
        return self._running

    @property
    def models_initialized(self) -> bool:
        """Whether models have been initialized."""
        return self._models_initialized

    @property
    def frame_count(self) -> int:
        """Total frames processed."""
        return self._frame_index
