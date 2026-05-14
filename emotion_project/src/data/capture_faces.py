"""Fallback face capturer — grabs labeled face crops from your webcam.

Usage:
    python src/data/capture_faces.py
You'll be prompted to pose with each emotion. The script auto-saves N frames
per emotion (default 50) with a Haar-cascade pre-crop so we don't store
huge full-frame images.

Controls during capture:
    Q  quit early
    Space  pause/resume
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.labels import EMOTIONS  # noqa: E402

CFG = load_config()
OUT_ROOT = resolve_path(CFG.paths.data_raw_faces)


def _load_face_detector() -> cv2.CascadeClassifier:
    """Use OpenCV's bundled frontal-face Haar cascade for speed during capture.
    (MTCNN runs in the proper preprocessing step.)
    """
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade at {cascade_path}")
    return detector


def _capture_for_emotion(cap: cv2.VideoCapture, detector, emotion: str, n_target: int) -> int:
    out_dir = OUT_ROOT / emotion
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    paused = False
    print(f"  Pose with a {emotion.upper()} expression. Capturing {n_target} crops...")

    while saved < n_target:
        ok, frame = cap.read()
        if not ok:
            print("  ✗ webcam read failed; aborting this emotion")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

        annotated = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        status = "PAUSED" if paused else f"{emotion}  {saved}/{n_target}"
        cv2.putText(
            annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
        cv2.imshow("Capture faces (Q=quit, Space=pause)", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return saved
        if key == ord(" "):
            paused = not paused
            time.sleep(0.2)
        if paused or len(faces) == 0:
            continue

        # Save the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        ts = int(time.time() * 1000)
        cv2.imwrite(str(out_dir / f"manual_{emotion.lower()}_{ts}_{saved}.jpg"), crop)
        saved += 1
        time.sleep(0.05)  # debounce

    return saved


def main() -> int:
    print("=== Webcam fallback face capturer ===")
    detector = _load_face_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open webcam.")
        return 1

    n_target = 50
    raw = input(f"Frames per emotion [{n_target}]: ").strip()
    if raw:
        n_target = int(raw)

    try:
        for emotion in EMOTIONS:
            input(f"\nPress <Enter> when ready for: {emotion}")
            saved = _capture_for_emotion(cap, detector, emotion, n_target)
            print(f"  ✓ {emotion}: {saved} crops saved")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n▶ NEXT: python src/data/validate_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
