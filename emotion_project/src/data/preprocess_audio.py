"""Phase 2 — audio preprocessing.

For every clip under data/raw/audio/<Emotion>/:
    1. Load with librosa, mono, resample to `audio.sample_rate`.
    2. Pad/trim to `audio.duration_seconds` exactly.
    3. Extract features via the shared `src.utils.audio_features` helper:
         • MFCC          (n_mfcc, T)
         • log-mel spec  (n_mels, T)
         • ZCR           (1, T)
         • log-F0 / voicing / intensity (3, T)   — parselmouth
         • jitter / shimmer / HNR (3, T)         — parselmouth (broadcast)
    4. Save all features + labels to data/processed/audio.pt.

The LSTM input is the time axis; features per timestep is the channel axis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.audio_features import expected_channels, extract_features  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.labels import EMOTIONS, LABEL_TO_INDEX  # noqa: E402

CFG = load_config()
SR = int(CFG.audio.sample_rate)
DUR = float(CFG.audio.duration_seconds)
N_MFCC = int(CFG.audio.n_mfcc)
N_MELS = int(CFG.audio.n_mels)
N_FFT = int(CFG.audio.n_fft)
HOP = int(CFG.audio.hop_length)
INCLUDE_PROSODY = True

RAW_AUDIO_DIR = resolve_path(CFG.paths.data_raw_audio)
OUT_PATH = resolve_path(CFG.paths.audio_tensor)
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}

TARGET_SAMPLES = int(SR * DUR)


def _load_and_fix_length(path: Path) -> np.ndarray | None:
    """Load mono audio at the configured sample rate, pad/trim to a fixed length."""
    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    except Exception:  # noqa: BLE001
        return None
    if y.size == 0:
        return None
    if y.size < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - y.size))
    else:
        y = y[:TARGET_SAMPLES]
    return y.astype(np.float32)


def _extract_features(y: np.ndarray) -> np.ndarray:
    return extract_features(
        y, sr=SR, n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP,
        include_prosody=INCLUDE_PROSODY,
    )


def main() -> int:
    print("=== Phase 2 — audio preprocessing ===")
    print(f"  sr={SR}Hz  dur={DUR}s  n_mfcc={N_MFCC}  n_mels={N_MELS}")
    print(f"  prosodic features: {'ON' if INCLUDE_PROSODY else 'OFF'}  "
          f"→ {expected_channels(N_MFCC, N_MELS, INCLUDE_PROSODY)} channels")

    feats: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    for emotion in EMOTIONS:
        d = RAW_AUDIO_DIR / emotion
        files = sorted(p for p in d.glob("*") if p.suffix.lower() in AUDIO_EXTS) if d.exists() else []
        print(f"  {emotion}: {len(files)} candidates")

        for path in tqdm(files, desc=emotion, leave=False):
            y = _load_and_fix_length(path)
            if y is None:
                skipped += 1
                continue
            try:
                f = _extract_features(y)
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
            feats.append(f)
            labels.append(LABEL_TO_INDEX[emotion])

    if not feats:
        print("✗ No audio features produced. Check data/raw/audio/ contents.")
        return 1

    # Pad time axes to a common length (should already be equal but guard against off-by-one)
    max_T = max(f.shape[1] for f in feats)
    C = feats[0].shape[0]
    arr = np.zeros((len(feats), C, max_T), dtype=np.float32)
    for i, f in enumerate(feats):
        arr[i, :, : f.shape[1]] = f

    tensor = torch.from_numpy(arr)  # (N, C, T)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": tensor,
            "labels": label_tensor,
            "n_mfcc": N_MFCC,
            "n_mels": N_MELS,
            "channels": C,
            "T": max_T,
            "sample_rate": SR,
            "duration": DUR,
        },
        OUT_PATH,
    )

    print(f"\n✓ Saved {tensor.shape[0]} audio feature tensors to {OUT_PATH}")
    print(f"  shape = {tuple(tensor.shape)}  skipped = {skipped}")
    print("▶ NEXT: python src/data/build_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
