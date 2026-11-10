from __future__ import annotations

import time
import cv2


def capture_webcam_sample(detector, camera_index: int = 0, duration_seconds: int = 5):
    """Capture a short webcam sample and return (last_frame, max_count, annotated_frame)."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {camera_index}. Close other camera apps, "
            "check OS camera permissions, or try camera index 1."
        )

    start = time.time()
    max_count = 0
    last_frame = None
    last_annotated = None

    try:
        while time.time() - start < duration_seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            count, annotated = detector.detect_frame(frame)
            max_count = max(max_count, count)
            last_frame = frame
            last_annotated = annotated
            time.sleep(0.02)
    finally:
        cap.release()

    if last_frame is None:
        raise RuntimeError("Webcam opened but no frames were received.")
    return last_frame, max_count, last_annotated
