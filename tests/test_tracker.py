"""
Tests for PRADAS Multi-Object Pedestrian Tracker.

Tests track creation, update, history bounds, simple IoU association fallback,
and stale track cleanup.
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import TrackingConfig
from src.tracker import TrackedPedestrian, PedestrianTracker, TrackHistory
from src.detector_pose import PedestrianDetection


class TestTrackedPedestrian:
    """Tests for the TrackedPedestrian dataclass and properties."""

    def test_pedestrian_properties(self):
        tp = TrackedPedestrian(
            track_id=1,
            bbox=[100.0, 50.0, 200.0, 250.0],
            confidence=0.9
        )
        assert tp.center == (150.0, 150.0)
        assert tp.height == 200.0
        assert tp.area == 20000.0

    def test_pedestrian_update(self):
        tp = TrackedPedestrian(
            track_id=1,
            bbox=[100.0, 50.0, 200.0, 250.0],
            confidence=0.9
        )
        tp.update(
            bbox=[110.0, 60.0, 210.0, 260.0],
            confidence=0.95,
            keypoints=None,
            frame_index=2,
            depth=0.7,
            timestamp=100.5
        )
        assert tp.frame_index == 2
        assert tp.track_age == 2
        assert tp.confidence == 0.95
        assert len(tp.history.centers) == 1
        assert tp.history.centers[0] == (160.0, 160.0)
        assert len(tp.history.depths) == 1
        assert tp.history.depths[0] == 0.7


class TestPedestrianTracker:
    """Tests for PedestrianTracker logic using the fallback assignment."""

    @pytest.fixture
    def tracker(self):
        config = TrackingConfig(track_buffer=5)
        # Use simple tracking by default (no model loaded)
        return PedestrianTracker(config)

    def test_new_detections_create_tracks(self, tracker):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            PedestrianDetection(bbox=[100, 100, 150, 250], confidence=0.85, frame_index=1),
            PedestrianDetection(bbox=[300, 100, 350, 250], confidence=0.90, frame_index=1),
        ]

        tracks = tracker.update(frame, detections, frame_index=1)
        assert len(tracks) == 2
        assert {t.track_id for t in tracks} == {1, 2}

    def test_tracks_updated_across_frames(self, tracker):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Frame 1
        dets1 = [
            PedestrianDetection(bbox=[100, 100, 150, 250], confidence=0.85, frame_index=1)
        ]
        tracks1 = tracker.update(frame, dets1, frame_index=1)
        assert len(tracks1) == 1
        tid = tracks1[0].track_id

        # Frame 2 with slight displacement (should match same track ID)
        dets2 = [
            PedestrianDetection(bbox=[102, 102, 152, 252], confidence=0.88, frame_index=2)
        ]
        tracks2 = tracker.update(frame, dets2, frame_index=2)
        assert len(tracks2) == 1
        assert tracks2[0].track_id == tid
        assert tracks2[0].track_age == 2

    def test_stale_tracks_expired(self, tracker):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets1 = [
            PedestrianDetection(bbox=[100, 100, 150, 250], confidence=0.85, frame_index=1)
        ]
        tracker.update(frame, dets1, frame_index=1)

        # Empty detections for frames 2 through 10 (buffer is 5 frames)
        for f in range(2, 10):
            active = tracker.update(frame, [], frame_index=f)

        assert len(active) == 0

    def test_empty_frame_and_empty_detections(self, tracker):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = tracker.update(frame, [], frame_index=1)
        assert tracks == []
