# PRADAS Models Directory

Model weights are downloaded automatically on first use and cached by the
respective frameworks (Ultralytics, PyTorch Hub).

## Models Used

### YOLOv8 Pose (yolov8n-pose.pt)
- **Purpose**: Pedestrian detection + pose estimation
- **Size**: ~6 MB (nano variant)
- **Downloaded by**: Ultralytics on first `YOLO("yolov8n-pose.pt")` call
- **Cache location**: `~/.cache/ultralytics/` or current directory

### MiDaS Small
- **Purpose**: Monocular relative depth estimation
- **Size**: ~25 MB
- **Downloaded by**: `torch.hub.load("intel-isl/MiDaS", "MiDaS_small")`
- **Cache location**: `~/.cache/torch/hub/`

## Larger Model Variants

For higher accuracy (at the cost of speed), you can use:

| Model | Config Key | Size | Speed |
|-------|-----------|------|-------|
| yolov8n-pose.pt | Default | ~6MB | Fastest |
| yolov8s-pose.pt | PRADAS_YOLO_POSE_MODEL | ~23MB | Fast |
| yolov8m-pose.pt | PRADAS_YOLO_POSE_MODEL | ~52MB | Medium |
| MiDaS_small | Default | ~25MB | Fastest |
| DPT_Hybrid | PRADAS_MIDAS_MODEL_TYPE | ~120MB | Medium |
| DPT_Large | PRADAS_MIDAS_MODEL_TYPE | ~470MB | Slow |

## Important

- Do NOT commit model weight files to the repository.
- Model files (`*.pt`, `*.pth`) are listed in `.gitignore`.
