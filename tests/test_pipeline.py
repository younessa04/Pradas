"""
Tests for PRADAS PedestrianRiskPipeline.
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import Settings
from src.pipeline import PedestrianRiskPipeline
from src.detector_pose import PedestrianDetection
from src.tracker import TrackedPedestrian, PedestrianTracker
from src.intent_analyzer import PedestrianIntentAnalyzer
from src.risk_ttc_engine import PedestrianRiskEngine
from src.visualization import FrameVisualizer
from utils.performance import PerformanceMonitor


class DummyDetector:
    is_initialized = True

    def detect(self, frame, frame_index=0):
        # Return a synthetic detection in the danger zone
        return [
            PedestrianDetection(
                bbox=[300.0, 300.0, 360.0, 420.0],
                confidence=0.92,
                frame_index=frame_index
            )
        ]


class TestPipelineFlow:
    """Tests for PedestrianRiskPipeline processing flow."""

    @pytest.fixture
    def mock_pipeline(self):
        settings = Settings()
        pipeline = PedestrianRiskPipeline(settings)

        # Inject components manually to avoid downloading neural network weights
        pipeline._perf_monitor = PerformanceMonitor()
        pipeline._detector = DummyDetector()
        pipeline._tracker = PedestrianTracker(settings.tracking)
        pipeline._intent_analyzer = PedestrianIntentAnalyzer(
            danger_zone_config=settings.danger_zone,
            frame_width=640,
            frame_height=480
        )
        pipeline._risk_engine = PedestrianRiskEngine(
            settings.risk, settings.danger_zone
        )
        pipeline._visualizer = FrameVisualizer()
        pipeline._models_initialized = True
        return pipeline

    def test_process_frame_end_to_end(self, mock_pipeline):
        events_emitted = []
        mock_pipeline.set_event_callback(lambda ev: events_emitted.append(ev))

        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Process a sequence of frames
        for f in range(1, 10):
            mock_pipeline._frame_index = f
            annotated, assessments, idx = mock_pipeline.process_frame(synthetic_frame)

            assert annotated.shape == (480, 640, 3)
            assert idx == f
            assert len(assessments) == 1
            assert assessments[0].track_id == 1

        # Check performance monitor recorded the frames
        assert mock_pipeline._perf_monitor.total_frames == 9
        assert mock_pipeline._perf_monitor.fps > 0

    def test_process_empty_frame(self, mock_pipeline):
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        annotated, assessments, idx = mock_pipeline.process_frame(empty)
        assert assessments == []
