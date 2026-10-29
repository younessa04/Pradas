"""
Tests for PRADAS Pedestrian Detection & Pose Estimation Data Structures.
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector_pose import PedestrianDetection, PedestrianDetectorPose, KEYPOINT_NAMES


class TestPedestrianDetection:
    """Tests for PedestrianDetection dataclass methods."""

    def test_bbox_properties(self):
        det = PedestrianDetection(
            bbox=[100.0, 50.0, 200.0, 250.0],
            confidence=0.85,
            frame_index=1
        )
        assert det.center == (150.0, 150.0)
        assert det.width == 100.0
        assert det.height == 200.0
        assert det.area == 20000.0

    def test_keypoint_access_and_centers(self):
        # 17 keypoints with (x, y, confidence)
        kps = np.zeros((17, 3), dtype=np.float32)
        # Left shoulder (index 5)
        kps[5] = [120.0, 80.0, 0.9]
        # Right shoulder (index 6)
        kps[6] = [160.0, 80.0, 0.85]
        # Left hip (index 11)
        kps[11] = [125.0, 150.0, 0.8]
        # Right hip (index 12)
        kps[12] = [155.0, 150.0, 0.8]

        det = PedestrianDetection(
            bbox=[100.0, 50.0, 200.0, 250.0],
            confidence=0.9,
            keypoints=kps
        )

        shoulder_kp = det.get_keypoint("left_shoulder")
        assert shoulder_kp is not None
        assert shoulder_kp[0] == 120.0
        assert shoulder_kp[1] == 80.0
        assert pytest.approx(shoulder_kp[2], 0.001) == 0.9

        shoulder_center = det.get_shoulder_center()
        assert shoulder_center == (140.0, 80.0)

        hip_center = det.get_hip_center()
        assert hip_center == (140.0, 150.0)

    def test_torso_orientation_facing_camera(self):
        kps = np.zeros((17, 3), dtype=np.float32)
        # Wide shoulders relative to bbox width 100 (40px > 0.3 * 100)
        kps[5] = [120.0, 80.0, 0.9]
        kps[6] = [160.0, 80.0, 0.85]

        det = PedestrianDetection(
            bbox=[100.0, 50.0, 200.0, 250.0],
            confidence=0.9,
            keypoints=kps
        )
        assert det.estimate_torso_orientation() == "facing_camera"

    def test_torso_orientation_profile(self):
        kps = np.zeros((17, 3), dtype=np.float32)
        # Very narrow shoulders (10px < 0.15 * 100)
        kps[5] = [140.0, 80.0, 0.9]
        kps[6] = [150.0, 80.0, 0.85]

        det = PedestrianDetection(
            bbox=[100.0, 50.0, 200.0, 250.0],
            confidence=0.9,
            keypoints=kps
        )
        # ls < rs -> facing_right
        assert det.estimate_torso_orientation() == "facing_right"


class TestDetectorPoseUnit:
    """Tests for PedestrianDetectorPose initialization guards."""

    def test_uninitialized_detector_returns_empty(self):
        detector = PedestrianDetectorPose()
        assert not detector.is_initialized
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert detector.detect(frame) == []
        assert detector.detect(None) == []
