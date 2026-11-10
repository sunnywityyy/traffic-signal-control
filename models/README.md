# Model files

This folder is intentionally shipped without binary model weights.

## Vehicle model
The default `yolov8n.pt` is downloaded by Ultralytics on first use. It detects COCO vehicle categories:
- car
- motorcycle
- bus
- truck

## Emergency vehicle model
A dedicated ambulance class is not part of the standard COCO label set. To enable automatic ambulance recognition:

1. Prepare a labelled dataset with an `ambulance` class (and optionally `fire_truck`, `police_vehicle`).
2. Train a YOLO detection model using Ultralytics.
3. Copy the resulting weights to `models/emergency_vehicle.pt`.
4. In the app, use that model path and set the class name to `ambulance`.

Example training command after preparing a YOLO-format dataset YAML:

```bash
yolo detect train model=yolov8n.pt data=your_emergency_dataset.yaml epochs=60 imgsz=640
```

Copy `runs/detect/train/weights/best.pt` to this folder as `emergency_vehicle.pt`.

Until then, use the manual demo trigger. It demonstrates the emergency preemption controller but is not automatic visual recognition.
