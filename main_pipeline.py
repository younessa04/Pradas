"""
PRADAS — Main Pipeline Entry Point.

Usage:
    python main_pipeline.py --source data/input/sample.mp4
    python main_pipeline.py --source 0                        # Webcam
    python main_pipeline.py --source data/input/sample.mp4 --headless
    python main_pipeline.py --source data/input/sample.mp4 --output data/output/result.mp4
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings, get_settings
from src.pipeline import PedestrianRiskPipeline
from utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PRADAS — Advanced Pedestrian Risk-Aware Driver Assistance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_pipeline.py --source data/input/sample.mp4
  python main_pipeline.py --source 0
  python main_pipeline.py --source data/input/video.mp4 --headless --output data/output/result.mp4
  python main_pipeline.py --source data/input/video.mp4 --max-frames 500

NOTE: This is a research prototype, not a certified ADAS.
      Risk estimates are approximate and must not be used for vehicle control.
        """
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: file path, camera index, or RTSP URL (default: 0 = webcam)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without display window"
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Output video file path (empty = no recording)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto)"
    )
    parser.add_argument(
        "--confidence", type=float, default=None,
        help="Detection confidence threshold (0-1, default: 0.30)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="Inference image resolution (e.g. 640, 960, 1280). Higher detects distant pedestrians much better"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="YOLO pose model path or name (default: yolov8n-pose.pt, e.g. yolov8s-pose.pt)"
    )
    parser.add_argument(
        "--frame-skip", type=int, default=None,
        help="Process every Nth frame (1 = no skip)"
    )
    parser.add_argument(
        "--night-mode", type=str, default=None,
        choices=["auto", "always", "never"],
        help="Night enhancement mode"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Configure settings from CLI args
    settings = get_settings()
    settings.device = args.device
    settings.log_level = args.log_level
    settings.video.source = args.source

    if args.output:
        settings.video.output_path = args.output
    if args.confidence is not None:
        settings.detection.confidence_threshold = args.confidence
        settings.tracking.track_high_thresh = args.confidence
        settings.tracking.new_track_thresh = args.confidence
    if args.imgsz is not None:
        settings.detection.imgsz = args.imgsz
    if args.model is not None:
        settings.detection.yolo_pose_model = args.model
    if args.frame_skip is not None:
        settings.video.frame_skip = args.frame_skip
    if args.night_mode is not None:
        settings.night_enhancement.mode = args.night_mode

    # Validate
    warnings = settings.validate()
    setup_logging(level=settings.log_level)
    for w in warnings:
        print(f"⚠ Config warning: {w}")

    # Create and run pipeline
    pipeline = PedestrianRiskPipeline(settings)

    print("\n" + "=" * 60)
    print("  PRADAS — Pedestrian Risk-Aware Driver Assistance System")
    print("  ⚠ Research Prototype — NOT a certified safety system")
    print("=" * 60 + "\n")

    pipeline.run(
        source=args.source,
        headless=args.headless,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
