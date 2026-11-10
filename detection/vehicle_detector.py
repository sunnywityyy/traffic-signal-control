from __future__ import annotations

import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


class VehicleDetector:
    """YOLO vehicle detector with a persistent tracker for one camera feed.

    Create one VehicleDetector instance per camera. Do not share an instance
    across camera feeds because ByteTrack maintains temporal tracking state.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect_frame_with_stats(self, frame):
        results = self.model.track(
            frame,
            conf=self.confidence,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            imgsz=640,
            device="cpu",
        )
        annotated = frame.copy()
        class_counts = {name: 0 for name in VEHICLE_CLASSES}
        if not results:
            return 0, annotated, class_counts

        result = results[0]
        names = result.names
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                class_name = str(names.get(cls_id, cls_id)).lower()
                if class_name not in VEHICLE_CLASSES:
                    continue
                class_counts[class_name] += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                track_id = int(box.id[0].item()) if box.id is not None else None
                label = f"{class_name} ID:{track_id}" if track_id is not None else class_name
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 220, 80), 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 80), 2)

        count = sum(class_counts.values())
        cv2.rectangle(annotated, (8, 8), (255, 48), (15, 20, 25), -1)
        cv2.putText(annotated, f"VEHICLES: {count}", (18, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)
        return count, annotated, class_counts

    def detect_frame(self, frame):
        count, annotated, _ = self.detect_frame_with_stats(frame)
        return count, annotated
