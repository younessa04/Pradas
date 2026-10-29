# PRADAS — Advanced Pedestrian Risk-Aware Driver Assistance System

> ⚠ **Research and Educational Prototype** — This is NOT a certified automotive safety system.
> Never use estimated risk or TTC as a guaranteed collision prediction.
> This system cannot control vehicle steering, braking, or acceleration.

## Overview

PRADAS is a modular AI prototype that analyzes road video (live or recorded) to:

1. **Detect pedestrians** using YOLOv8 with pose estimation
2. **Track pedestrians** persistently across frames via ByteTrack
3. **Estimate relative depth** using MiDaS monocular depth estimation
4. **Enhance low-light** video frames automatically
5. **Analyze pedestrian intent** (movement direction, crossing behavior)
6. **Assess collision risk** with configurable danger zones and TTC estimation
7. **Visualize results** with annotated video output
8. **Stream alerts** via FastAPI REST/WebSocket for mobile (Flutter) integration

## Architecture

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│  Video      │──▶│  Vision      │──▶│  YOLO       │
│  Source     │   │  Enhancer    │   │  Detector   │
└─────────────┘   └──────────────┘   └─────┬───────┘
                                            │
                        ┌───────────────────┘
                        ▼
                  ┌─────────────┐   ┌──────────────┐
                  │  ByteTrack  │──▶│  MiDaS       │
                  │  Tracker    │   │  Depth       │
                  └─────┬───────┘   └─────┬────────┘
                        │                 │
                        ▼                 ▼
                  ┌─────────────┐   ┌──────────────┐
                  │  Intent     │──▶│  Risk/TTC    │
                  │  Analyzer   │   │  Engine      │
                  └─────────────┘   └─────┬────────┘
                                          │
                    ┌─────────────────────┘
                    ▼                 ▼
              ┌───────────┐   ┌──────────────┐
              │ Visualizer│   │  WebSocket   │
              │ (Output)  │   │  API         │
              └───────────┘   └──────────────┘
```

## Project Structure

```
PRADAS/
├── api/                    # FastAPI REST + WebSocket server
│   ├── server.py           # API endpoints
│   ├── schemas.py          # Pydantic request/response models
│   └── websocket_manager.py # WebSocket client management
├── config/
│   └── settings.py         # Centralized configuration
├── data/
│   ├── input/              # Input videos (not tracked)
│   ├── output/             # Output videos (not tracked)
│   └── samples/            # Sample data
├── models/                 # Model weights (auto-downloaded)
├── src/                    # Core processing modules
│   ├── vision_enhancer.py  # Low-light enhancement
│   ├── detector_pose.py    # YOLOv8 detection + pose
│   ├── tracker.py          # ByteTrack pedestrian tracking
│   ├── depth_estimator.py  # MiDaS depth estimation
│   ├── intent_analyzer.py  # Movement/crossing analysis
│   ├── risk_ttc_engine.py  # Risk assessment + TTC
│   ├── visualization.py    # Frame annotation
│   └── pipeline.py         # Central orchestrator
├── utils/                  # Shared utilities
│   ├── geometry.py         # Point-in-polygon, bbox ops
│   ├── logging_config.py   # Structured logging
│   └── performance.py      # FPS/latency monitoring
├── tests/                  # Automated tests
├── main_pipeline.py        # CLI entry point
├── run_server.py           # API server entry point
├── requirements.txt        # Python dependencies
├── .env.example            # Configuration template
└── README.md               # This file
```

## System Requirements

- **Python**: 3.11 or 3.12
- **OS**: Windows 10/11, Linux, or macOS
- **RAM**: 4+ GB recommended
- **GPU**: Optional (NVIDIA CUDA for faster inference)
- **Webcam**: Optional (for live video)

## Installation

### 1. Clone or navigate to the project

```powershell
cd "path\to\PRADAS"
```

### 2. Create a virtual environment (recommended)

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies (CPU)

```powershell
pip install -r requirements.txt
```

### 4. Install dependencies (CUDA — for NVIDIA GPU)

If you have an NVIDIA GPU and want GPU acceleration:

```powershell
# Uninstall CPU torch first
pip uninstall torch torchvision -y

# Install CUDA-enabled PyTorch (check https://pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install remaining dependencies
pip install -r requirements.txt
```

### 5. Copy environment configuration

```powershell
copy .env.example .env
# Edit .env to customize settings
```

## Usage

### Run with a video file

```powershell
python main_pipeline.py --source data/input/sample.mp4
```

### Run with webcam

```powershell
python main_pipeline.py --source 0
```

### Run headless (no display window)

```powershell
python main_pipeline.py --source data/input/sample.mp4 --headless
```

### Save output video

```powershell
python main_pipeline.py --source data/input/sample.mp4 --output data/output/result.mp4
```

### Adjust detection confidence

```powershell
python main_pipeline.py --source 0 --confidence 0.6
```

### Enable night enhancement

```powershell
python main_pipeline.py --source data/input/night_video.mp4 --night-mode always
```

### Process every 2nd frame (faster, lower accuracy)

```powershell
python main_pipeline.py --source data/input/sample.mp4 --frame-skip 2
```

### Start the API server

```powershell
python run_server.py
# Server starts at http://0.0.0.0:8000
# WebSocket at ws://0.0.0.0:8000/ws/alerts
```

### Run tests

```powershell
python -m pytest tests/ -v
```

## Model Downloads

Models are downloaded automatically on first use:

| Model | Size | First-run download |
|-------|------|-------------------|
| yolov8n-pose.pt | ~6 MB | Via Ultralytics |
| MiDaS_small | ~25 MB | Via PyTorch Hub |

For larger/more accurate models, set environment variables:

```
PRADAS_YOLO_POSE_MODEL=yolov8s-pose.pt
PRADAS_MIDAS_MODEL_TYPE=DPT_Hybrid
```

## Configuration

All settings can be configured via environment variables (prefix `PRADAS_`)
or by editing the `.env` file. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `PRADAS_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `PRADAS_CONFIDENCE_THRESHOLD` | `0.5` | Detection confidence [0-1] |
| `PRADAS_NIGHT_MODE` | `auto` | `auto`, `always`, `never` |
| `PRADAS_FRAME_SKIP` | `1` | Process every Nth frame |
| `PRADAS_VIDEO_SOURCE` | `0` | File path or camera index |
| `PRADAS_API_HOST` | `0.0.0.0` | API server host |
| `PRADAS_API_PORT` | `8000` | API server port |
| `PRADAS_WS_ENABLED` | `false` | Enable WebSocket publishing |

## Danger Zone Configuration

The danger zone is defined as a normalized polygon (coordinates in [0, 1]).
Default: a trapezoidal region in front of the vehicle.

Edit in `config/settings.py` or extend to load from config file:

```python
DangerZoneConfig(
    polygon_normalized=[
        (0.2, 0.5),   # top-left
        (0.8, 0.5),   # top-right
        (0.95, 1.0),  # bottom-right
        (0.05, 1.0),  # bottom-left
    ]
)
```

**Note:** This is an image-space polygon — it approximates the vehicle's
path only for a fixed camera position and road geometry.

## Risk Score Interpretation

| Level | Score Range | Meaning |
|-------|------------|---------|
| LOW | 0.0 – 0.4 | Pedestrian detected, low interaction risk |
| MEDIUM | 0.4 – 0.7 | Approaching danger zone or uncertain data |
| HIGH | 0.7 – 1.0 | Evidence suggests possible collision path |

Risk scores combine: danger zone proximity, depth trend, TTC estimate,
intent classification, and detection confidence using weighted scoring.

## TTC (Time-to-Collision) Limitations

Three TTC methods are implemented, each with explicit limitations:

1. **Bounding-box expansion**: Approximate, image-plane only. NOT physical TTC.
   Confidence capped at 0.6.
2. **Depth trend**: Uses relative MiDaS depth. Very rough ordinal estimate.
   Confidence capped at 0.4.
3. **Calibrated metric**: Requires explicit camera calibration (not provided
   by default). Returns `null` without calibration.

**TTC values marked with `*` in the display are approximate indicators,
NOT calibrated physical predictions.**

## Metric Distance Calibration

MiDaS produces **relative depth** (higher = closer), NOT metric distance.

To get metric distances:
1. Calibrate your camera setup
2. Call `depth_estimator.set_calibration(scale, offset, unit="meters")`
3. Only then will `distance_meters` fields be populated

**Without calibration, `distance_meters` is always `null`.** The system
will never fabricate metric distances from uncalibrated relative depth.

## REST API

### GET /status

```json
{
  "status": "running",
  "api_version": "1.0.0",
  "models_initialized": true,
  "active_ws_clients": 2,
  "pipeline_status": "running",
  "device": "cpu"
}
```

### GET /health

```json
{
  "healthy": true,
  "api_up": true,
  "models_ready": true,
  "pipeline_running": true
}
```

## WebSocket Integration

### Connect

```
ws://localhost:8000/ws/alerts
```

### Message Format

```json
{
  "schema_version": "1.0",
  "event_type": "pedestrian_risk_update",
  "timestamp": "2026-10-24T12:00:00Z",
  "frame_index": 120,
  "pedestrians": [
    {
      "track_id": 7,
      "risk_level": "HIGH",
      "risk_score": 0.87,
      "ttc_seconds": 1.8,
      "ttc_method": "bbox_expansion",
      "ttc_confidence": 0.55,
      "relative_depth": 0.61,
      "distance_meters": null,
      "intent_state": "POSSIBLE_CROSSING",
      "danger_zone_status": "INSIDE",
      "reason_codes": ["INSIDE_DANGER_ZONE", "SHORT_TTC"]
    }
  ]
}
```

## Flutter Integration

To integrate with a Flutter app:

1. Add `web_socket_channel` package to your Flutter project
2. Connect to `ws://your-server:8000/ws/alerts`
3. Parse incoming JSON using the schema above
4. Handle `pedestrian_risk_update` events to display alerts

```dart
import 'package:web_socket_channel/web_socket_channel.dart';
import 'dart:convert';

final channel = WebSocketChannel.connect(
  Uri.parse('ws://192.168.1.x:8000/ws/alerts'),
);

channel.stream.listen((message) {
  final data = jsonDecode(message);
  if (data['event_type'] == 'pedestrian_risk_update') {
    for (var ped in data['pedestrians']) {
      if (ped['risk_level'] == 'HIGH') {
        // Show alert to driver
      }
    }
  }
});
```

## Running Tests

```powershell
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_geometry.py -v

# With coverage
python -m pytest tests/ -v --tb=short
```

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `ModuleNotFoundError: ultralytics` | Run `pip install ultralytics` |
| `CUDA not available` | Install CUDA-enabled PyTorch (see installation) |
| `Model download fails` | Check internet connection; models need first-run download |
| `Camera not opening` | Check camera index; try `--source 1` |
| `cv2.error: display` | Use `--headless` flag |
| `WebSocket timeout` | Check firewall; ensure server is running |
| `Out of memory` | Use smaller models or `--frame-skip 2` |

## Known Limitations

1. **Not a certified ADAS** — Cannot control vehicle systems
2. **MiDaS depth is relative** — Not metric distance without calibration
3. **TTC is approximate** — Bounding-box expansion TTC is not physical TTC
4. **Pose ≠ Intent** — Keypoints estimate posture, not pedestrian intentions
5. **Image-plane analysis** — All direction estimates are in image coordinates
6. **Single camera** — No stereo depth or multi-view fusion
7. **No vehicle speed input** — TTC does not account for ego-vehicle speed
8. **Track IDs are temporary** — May change after occlusion or tracker reset
9. **No audio warnings** — Alert output is visual and JSON only
10. **CPU performance** — Real-time processing requires GPU for complex scenes

## Future Extensions

The architecture supports future additions:

- Flutter real-time dashboard
- Historical risk-event storage (SQLite/PostgreSQL)
- Audio/push notifications
- Camera calibration + ground-plane estimation
- Vehicle speed integration for metric TTC
- Trajectory prediction (LSTM/Transformer)
- Docker deployment
- Privacy-aware video retention
- CSV/JSON risk reports export

## License

This project is for educational and research purposes.
