"""
PRADAS Performance Monitoring Utilities.

FPS tracking, per-stage latency, and performance reporting.
"""

import time
from collections import deque
from typing import Dict, Optional
from dataclasses import dataclass, field

from utils.logging_config import get_logger

logger = get_logger("performance")


@dataclass
class StageTimer:
    """Tracks latency for a single processing stage."""
    name: str
    _start_time: float = 0.0
    _durations: deque = field(default_factory=lambda: deque(maxlen=100))

    def start(self) -> None:
        """Mark start of stage."""
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Mark end of stage, return duration in ms."""
        duration_ms = (time.perf_counter() - self._start_time) * 1000.0
        self._durations.append(duration_ms)
        return duration_ms

    @property
    def avg_ms(self) -> float:
        """Average duration in milliseconds."""
        if not self._durations:
            return 0.0
        return sum(self._durations) / len(self._durations)

    @property
    def last_ms(self) -> float:
        """Last duration in milliseconds."""
        return self._durations[-1] if self._durations else 0.0


class PerformanceMonitor:
    """
    Monitors FPS and per-stage latency for the processing pipeline.

    Usage:
        monitor = PerformanceMonitor()
        monitor.start_frame()
        with monitor.stage("detection"):
            # ... detection code ...
        with monitor.stage("tracking"):
            # ... tracking code ...
        monitor.end_frame()
        print(f"FPS: {monitor.fps:.1f}")
    """

    def __init__(self, window_size: int = 60):
        """
        Args:
            window_size: Number of recent frames for averaging.
        """
        self._stages: Dict[str, StageTimer] = {}
        self._frame_times: deque = deque(maxlen=window_size)
        self._frame_start: float = 0.0
        self._total_frames: int = 0
        self._start_time: float = time.perf_counter()

    def start_frame(self) -> None:
        """Mark the start of frame processing."""
        self._frame_start = time.perf_counter()

    def end_frame(self) -> float:
        """
        Mark the end of frame processing.

        Returns:
            Frame processing time in milliseconds.
        """
        frame_time = (time.perf_counter() - self._frame_start) * 1000.0
        self._frame_times.append(frame_time)
        self._total_frames += 1
        return frame_time

    class _StageContext:
        """Context manager for timing a processing stage."""

        def __init__(self, timer: StageTimer):
            self._timer = timer

        def __enter__(self) -> StageTimer:
            self._timer.start()
            return self._timer

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            self._timer.stop()

    def stage(self, name: str) -> _StageContext:
        """
        Create a context manager for timing a named stage.

        Args:
            name: Stage name (e.g., "detection", "tracking").

        Returns:
            Context manager that times the enclosed code.
        """
        if name not in self._stages:
            self._stages[name] = StageTimer(name=name)
        return self._StageContext(self._stages[name])

    @property
    def fps(self) -> float:
        """Current frames per second (averaged over recent frames)."""
        if len(self._frame_times) < 2:
            return 0.0
        avg_ms = sum(self._frame_times) / len(self._frame_times)
        if avg_ms <= 0:
            return 0.0
        return 1000.0 / avg_ms

    @property
    def frame_time_ms(self) -> float:
        """Last frame processing time in ms."""
        return self._frame_times[-1] if self._frame_times else 0.0

    @property
    def total_frames(self) -> int:
        """Total frames processed."""
        return self._total_frames

    @property
    def uptime_seconds(self) -> float:
        """Total uptime in seconds."""
        return time.perf_counter() - self._start_time

    def get_stage_report(self) -> Dict[str, float]:
        """
        Get average latency for each stage.

        Returns:
            Dict mapping stage name to average ms.
        """
        return {name: timer.avg_ms for name, timer in self._stages.items()}

    def get_summary(self) -> str:
        """Get a human-readable performance summary."""
        lines = [
            f"FPS: {self.fps:.1f}",
            f"Total Frames: {self._total_frames}",
            f"Uptime: {self.uptime_seconds:.1f}s",
            "Stage Latencies (avg ms):"
        ]
        for name, timer in self._stages.items():
            lines.append(f"  {name}: {timer.avg_ms:.1f}ms")
        return "\n".join(lines)

    def log_summary(self, interval_frames: int = 100) -> None:
        """Log performance summary periodically."""
        if self._total_frames > 0 and self._total_frames % interval_frames == 0:
            logger.info(f"Performance @ frame {self._total_frames}: "
                        f"FPS={self.fps:.1f}, "
                        f"frame_time={self.frame_time_ms:.1f}ms")
            for name, timer in self._stages.items():
                logger.debug(f"  Stage '{name}': avg={timer.avg_ms:.1f}ms, "
                             f"last={timer.last_ms:.1f}ms")
