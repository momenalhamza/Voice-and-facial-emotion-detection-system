"""Phase 2 — build stratified train/val/test splits.

Produces JSON manifests under data/processed/splits/ describing which sample
indices belong to each split, for each modality:

    splits/face_train.json   {"indices":[...], "labels":[...]}
    splits/face_val.json
    splits/face_test.json
    splits/audio_train.json
    splits/audio_val.json
    splits/audio_test.json

The fusion model uses both manifests independently — it does not require
paired face/audio samples, since the public datasets (RAVDESS speech,
FER2013 images) don't ship paired data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.labels import EMOTIONS  # noqa: E402

CFG = load_config()
SPLITS_DIR = resolve_path(CFG.paths.splits_dir)
FACES_PATH = resolve_path(CFG.paths.faces_tensor)
AUDIO_PATH = resolve_path(CFG.paths.audio_tensor)
TRAIN_R = float(CFG.split.train_ratio)
VAL_R = float(CFG.split.val_ratio)
TEST_R = float(CFG.split.test_ratio)
SEED = int(CFG.project.seed)


def _stratified_three_way(labels: np.ndarray):
    """Stratified 70/15/15 split implemented as two consecutive splits."""
    idx = np.arange(labels.shape[0])
    train_idx, temp_idx, _, temp_labels = train_test_split(
        idx, labels, test_size=(VAL_R + TEST_R), stratify=labels, random_state=SEED
    )
    rel_val = VAL_R / (VAL_R + TEST_R)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, test_size=(1 - rel_val), stratify=temp_labels, random_state=SEED
    )
    return train_idx, val_idx, test_idx


def _save_manifest(name: str, indices: np.ndarray, labels: np.ndarray) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out = SPLITS_DIR / f"{name}.json"
    out.write_text(json.dumps({
        "indices": indices.tolist(),
        "labels": labels[indices].tolist(),
    }))
    counts = np.bincount(labels[indices], minlength=len(EMOTIONS))
    counts_str = "  ".join(f"{EMOTIONS[i]}:{int(counts[i])}" for i in range(len(EMOTIONS)))
    print(f"  {out.name}: {len(indices)} samples   ({counts_str})")


def _split_one(modality: str, blob_path: Path, label_key: str) -> None:
    if not blob_path.exists():
        print(f"  ✗ missing {blob_path} — skip {modality}")
        return
    blob = torch.load(blob_path, map_location="cpu", weights_only=False)
    labels = blob[label_key].numpy() if hasattr(blob[label_key], "numpy") else np.asarray(blob[label_key])
    train_idx, val_idx, test_idx = _stratified_three_way(labels)
    print(f"\n[{modality}]  total={labels.shape[0]}")
    _save_manifest(f"{modality}_train", train_idx, labels)
    _save_manifest(f"{modality}_val", val_idx, labels)
    _save_manifest(f"{modality}_test", test_idx, labels)


def main() -> int:
    assert abs(TRAIN_R + VAL_R + TEST_R - 1.0) < 1e-6, "split ratios must sum to 1"
    print("=== Phase 2 — build stratified splits ===")
    print(f"  ratios: train={TRAIN_R} val={VAL_R} test={TEST_R}  seed={SEED}")

    _split_one("face", FACES_PATH, "labels")
    _split_one("audio", AUDIO_PATH, "labels")

    print("\n✓ Splits written to", SPLITS_DIR)
    print("▶ NEXT: python train_face.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
