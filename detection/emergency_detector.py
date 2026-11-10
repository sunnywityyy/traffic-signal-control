from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO


class EmergencyDetector:
    """Optional custom YOLO detector. Requires weights trained with an ambulance class."""

    def __init__(self, model_path: str, emergency_class_name: str = "ambulance", confidence: float = 0.45):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Custom emergency model not found: {path}. "
                "See models/README.md. Use the manual demo trigger until custom weights are installed."
            )
        self.model = YOLO(str(path))
        self.class_name = emergency_class_name.strip().lower()
        self.confidence = confidence

    def analyze_video(self, video_path: str, max_frames: int = 500, confirmation_hits: int = 3):
        """Return (detected, frames_checked, best_confidence)."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frames_checked = 0
        hits = 0
        best_conf = 0.0
        try:
            while frames_checked < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_checked += 1
                results = self.model.predict(frame, conf=self.confidence, verbose=False)
                if not results or results[0].boxes is None:
                    continue
                result = results[0]
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    detected_name = str(result.names.get(class_id, class_id)).lower()
                    conf = float(box.conf[0].item())
                    if self.class_name in detected_name and conf >= self.confidence:
                        hits += 1
                        best_conf = max(best_conf, conf)
                        break
                if hits >= confirmation_hits:
                    return True, frames_checked, best_conf
        finally:
            cap.release()

        return False, frames_checked, best_conf
