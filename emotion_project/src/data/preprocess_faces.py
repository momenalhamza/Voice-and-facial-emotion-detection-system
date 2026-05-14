"""Phase 2 — face preprocessing.

For every image under data/raw/faces/<Emotion>/:
    1. Read with OpenCV (or PIL fallback for grayscale).
    2. Detect the largest face with MTCNN (falls back to Haar cascade if MTCNN
       fails — useful when MTCNN can't load weights offline).
    3. Crop, resize to `face.image_size` (default 48), convert to 3-channel.
    4. Normalize to float32 in [0,1].
    5. Stack into one tensor + label tensor and save to data/processed/faces.pt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.labels import EMOTIONS, LABEL_TO_INDEX  # noqa: E402

CFG = load_config()
IMAGE_SIZE = int(CFG.face.image_size)
RAW_FACES_DIR = resolve_path(CFG.paths.data_raw_faces)
OUT_PATH = resolve_path(CFG.paths.faces_tensor)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- #
# Detectors                                                                   #
# --------------------------------------------------------------------------- #
def _load_mtcnn():
    """Return an MTCNN detector or None if facenet-pytorch is unavailable."""
    try:
        from facenet_pytorch import MTCNN  # type: ignore
    except ImportError:
        print("facenet-pytorch not installed; falling back to Haar cascade.")
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MTCNN(image_size=IMAGE_SIZE, margin=10, post_process=False, device=device)


def _load_haar() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(str(cascade_path))


def _detect_with_mtcnn(mtcnn, img_rgb: np.ndarray) -> np.ndarray | None:
    """Return a 48×48×3 RGB uint8 crop or None if no face is detected."""
    try:
        face = mtcnn(img_rgb)
    except Exception:  # noqa: BLE001
        return None
    if face is None:
        return None
    # MTCNN returns CHW float tensor in [0,255] when post_process=False.
    arr = face.permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.resize(arr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def _detect_with_haar(detector: cv2.CascadeClassifier, img_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    if len(faces) == 0:
        # No face → just use the whole image. Many FER2013 crops already are face-only.
        crop = img_bgr
    else:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop = img_bgr[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=== Phase 2 — face preprocessing ===")
    mtcnn = _load_mtcnn()
    haar = _load_haar()

    images: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    for emotion in EMOTIONS:
        d = RAW_FACES_DIR / emotion
        files = sorted(p for p in d.glob("*") if p.suffix.lower() in IMG_EXTS) if d.exists() else []
        print(f"  {emotion}: {len(files)} candidates")

        for img_path in tqdm(files, desc=emotion, leave=False):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                skipped += 1
                continue
            if img_bgr.ndim == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

            crop_rgb = None
            if mtcnn is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                crop_rgb = _detect_with_mtcnn(mtcnn, img_rgb)
            if crop_rgb is None:
                crop_rgb = _detect_with_haar(haar, img_bgr)
            if crop_rgb is None:
                skipped += 1
                continue

            images.append(crop_rgb)
            labels.append(LABEL_TO_INDEX[emotion])

    if not images:
        print("✗ No face crops produced. Check data/raw/faces/ contents.")
        return 1

    # Stack: (N, H, W, 3) → (N, 3, H, W) float32 in [0,1]
    arr = np.stack(images).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    label_tensor = torch.tensor(labels, dtype=torch.long)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": tensor, "labels": label_tensor, "size": IMAGE_SIZE}, OUT_PATH)

    print(f"\n✓ Saved {tensor.shape[0]} face crops to {OUT_PATH}")
    print(f"  shape = {tuple(tensor.shape)}  skipped = {skipped}")
    print("▶ NEXT: python src/data/preprocess_audio.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
