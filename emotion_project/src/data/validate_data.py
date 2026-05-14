"""Sanity-check what's on disk after Phase 1.

Reports per-emotion counts for both modalities and warns about class imbalance
or empty buckets. Exit code is non-zero if any emotion has zero samples.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.labels import EMOTIONS  # noqa: E402

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _count(root: Path, exts: set[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for emotion in EMOTIONS:
        d = root / emotion
        if not d.exists():
            out[emotion] = 0
            continue
        out[emotion] = sum(1 for p in d.iterdir() if p.suffix.lower() in exts)
    return out


def _print_table(title: str, counts: dict[str, int]) -> None:
    print(f"\n{title}")
    print("-" * (len(title)))
    width = max(len(e) for e in EMOTIONS)
    total = 0
    for emotion in EMOTIONS:
        n = counts[emotion]
        total += n
        bar = "█" * min(40, n // max(1, max(counts.values()) // 40 or 1))
        print(f"  {emotion.ljust(width)}  {n:>6}  {bar}")
    print(f"  {'TOTAL'.ljust(width)}  {total:>6}")


def main() -> int:
    cfg = load_config()
    audio_root = resolve_path(cfg.paths.data_raw_audio)
    faces_root = resolve_path(cfg.paths.data_raw_faces)

    audio_counts = _count(audio_root, AUDIO_EXTS)
    face_counts = _count(faces_root, IMAGE_EXTS)

    _print_table(f"Audio (data/raw/audio)", audio_counts)
    _print_table(f"Faces (data/raw/faces)", face_counts)

    empty_audio = [e for e, n in audio_counts.items() if n == 0]
    empty_faces = [e for e, n in face_counts.items() if n == 0]
    if empty_audio:
        print(f"\n✗ Empty audio buckets: {', '.join(empty_audio)}")
    if empty_faces:
        print(f"✗ Empty face buckets:  {', '.join(empty_faces)}")
    if empty_audio or empty_faces:
        print("\nFix the empty buckets before moving on (re-run download_data.py "
              "or use record_audio.py / capture_faces.py).")
        return 1

    # Mild imbalance warning
    def _imbalance(counts: dict[str, int]) -> float:
        vals = list(counts.values())
        return max(vals) / max(1, min(vals))

    if _imbalance(audio_counts) > 3:
        print("\n⚠ Audio class imbalance > 3×. Consider trimming or augmenting.")
    if _imbalance(face_counts) > 3:
        print("⚠ Face class imbalance > 3×. Consider trimming or augmenting.")

    print("\n✓ Validation passed.")
    print("▶ NEXT: python src/data/preprocess_faces.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
