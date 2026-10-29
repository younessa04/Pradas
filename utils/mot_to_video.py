"""
PRADAS — MOT17 Dataset to MP4 Converter Utility.

Converts MOT17 image sequences into standard MP4 videos ready for PRADAS.

Usage:
    python utils/mot_to_video.py --seq data/MOT17/train/MOT17-13-FRCNN
    python utils/mot_to_video.py --all-train
    python utils/mot_to_video.py --seq data/MOT17/train/MOT17-04-FRCNN --output data/input/MOT17-04-night.mp4
"""

import os
import sys
import glob
import configparser
import argparse
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.logging_config import get_logger

logger = get_logger("mot_to_video")


def convert_sequence_to_video(seq_dir: str, output_path: str = None, fps: float = None) -> str:
    """
    Convert a single MOT17 sequence directory to MP4 video.

    Args:
        seq_dir: Path to MOT17 sequence directory (e.g., data/MOT17/train/MOT17-13-FRCNN).
        output_path: Target video file path. Defaults to data/input/<seq_name>.mp4.
        fps: Target framerate. If None, read from seqinfo.ini (default 25/30).

    Returns:
        Path to generated video file.
    """
    if not os.path.exists(seq_dir):
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

    # Locate image directory
    img_dir = os.path.join(seq_dir, "img1")
    if not os.path.exists(img_dir):
        img_dir = seq_dir

    # Find all jpg/png frames
    frames = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) or glob.glob(os.path.join(img_dir, "*.png")))
    if not frames:
        raise ValueError(f"No image frames found in {img_dir}")

    # Read sequence info if available
    seq_name = os.path.basename(os.path.normpath(seq_dir))
    seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")
    target_fps = fps

    if target_fps is None and os.path.exists(seqinfo_path):
        config = configparser.ConfigParser()
        config.read(seqinfo_path)
        if "Sequence" in config and "frameRate" in config["Sequence"]:
            try:
                target_fps = float(config["Sequence"]["frameRate"])
            except ValueError:
                target_fps = 25.0
    if target_fps is None:
        target_fps = 25.0

    # Default output path
    if output_path is None:
        os.makedirs("data/input", exist_ok=True)
        # Simplify name, e.g. MOT17-13-FRCNN -> MOT17-13.mp4
        short_name = seq_name.split("-FRCNN")[0].split("-DPM")[0].split("-SDP")[0]
        output_path = os.path.join("data", "input", f"{short_name}.mp4")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Read first frame for dimensions
    first_frame = cv2.imread(frames[0])
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frames[0]}")

    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, (w, h))

    logger.info(f"Converting {len(frames)} frames from {seq_name} to {output_path} ({w}x{h} @ {target_fps} FPS)...")

    for idx, fpath in enumerate(frames):
        frame = cv2.imread(fpath)
        if frame is not None:
            writer.write(frame)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(frames):
            logger.info(f"  Processed {idx + 1}/{len(frames)} frames...")

    writer.release()
    logger.info(f"✓ Video successfully created: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert MOT17 sequences to MP4 video")
    parser.add_argument("--seq", type=str, help="Path to single MOT17 sequence directory")
    parser.add_argument("--output", type=str, default=None, help="Output MP4 path")
    parser.add_argument("--fps", type=float, default=None, help="Target framerate override")
    parser.add_argument("--all-train", action="store_true", help="Convert all unique train sequences")
    args = parser.parse_args()

    if args.all_train:
        train_dir = os.path.join("data", "MOT17", "train")
        # Only take FRCNN variants to avoid 3x duplicate conversion
        seqs = sorted(glob.glob(os.path.join(train_dir, "*FRCNN*")))
        for seq in seqs:
            convert_sequence_to_video(seq)
    elif args.seq:
        convert_sequence_to_video(args.seq, args.output, args.fps)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
