"""
PRADAS — API Server Entry Point.

Usage:
    python run_server.py
    python run_server.py --host 0.0.0.0 --port 8000
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

from api.server import create_app
from utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PRADAS API Server"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level", type=str, default="info",
        help="Uvicorn log level"
    )
    return parser.parse_args()


def main() -> None:
    """Start the API server."""
    args = parse_args()
    setup_logging(level=args.log_level.upper())

    print("\n" + "=" * 50)
    print("  PRADAS API Server")
    print(f"  http://{args.host}:{args.port}")
    print(f"  WebSocket: ws://{args.host}:{args.port}/ws/alerts")
    print(f"  Status: http://{args.host}:{args.port}/status")
    print("=" * 50 + "\n")

    app = create_app()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
