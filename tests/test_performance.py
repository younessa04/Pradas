"""
Tests for PRADAS Performance Monitoring Utilities.
"""

import time
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.performance import PerformanceMonitor, StageTimer


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor and StageTimer."""

    def test_stage_timer_measures_duration(self):
        timer = StageTimer(name="test_stage")
        timer.start()
        time.sleep(0.01)  # 10ms
        elapsed = timer.stop()

        assert elapsed >= 8.0  # at least ~8ms
        assert timer.avg_ms >= 8.0
        assert timer.last_ms >= 8.0

    def test_performance_monitor_fps_and_stages(self):
        monitor = PerformanceMonitor(window_size=10)

        for _ in range(5):
            monitor.start_frame()
            with monitor.stage("detection"):
                time.sleep(0.005)
            with monitor.stage("tracking"):
                time.sleep(0.002)
            frame_ms = monitor.end_frame()
            assert frame_ms > 0

        assert monitor.total_frames == 5
        assert monitor.fps > 0
        report = monitor.get_stage_report()
        assert "detection" in report
        assert "tracking" in report
        assert report["detection"] > 0
        assert report["tracking"] > 0

        summary = monitor.get_summary()
        assert "FPS:" in summary
        assert "detection:" in summary
