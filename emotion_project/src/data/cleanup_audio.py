"""One-shot cleaner for the RAVDESS download.

The initial labels mapping incorrectly bucketed RAVDESS emotion codes 07
(disgust) and 08 (surprised) into 'Neutral'. This script walks each
emotion folder and removes any file whose filename indicates it is one of
those dropped emotions, so the on-disk distribution matches the corrected
label policy.

Run once after upgrading labels.py:
    python src/data/cleanup_audio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.labels import EMOTIONS  # noqa: E402

DROP_CODES = {"07", "08"}  # RAVDESS disgust + surprised


def _is_dropped_ravdess(name: str) -> bool:
    parts = name.split("-")
    return len(parts) >= 3 and parts[2] in DROP_CODES


def main() -> int:
    cfg = load_config()
    audio_root = resolve_path(cfg.paths.data_raw_audio)
    removed = 0
    for emotion in EMOTIONS:
        d = audio_root / emotion
        if not d.exists():
            continue
        for wav in d.glob("*.wav"):
            if _is_dropped_ravdess(wav.stem):
                wav.unlink()
                removed += 1
    print(f"✓ removed {removed} mis-labeled RAVDESS wavs (codes 07 + 08)")
    print("▶ NEXT: python src/data/validate_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
