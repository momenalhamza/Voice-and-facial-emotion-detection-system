"""Fallback recorder — capture your own labeled audio samples from the mic.

Usage:
    python src/data/record_audio.py
You'll be prompted to choose an emotion and the number of takes per emotion.

Each take is N seconds (default 3s) of 16 kHz mono PCM, saved as a .wav into
data/raw/audio/<Emotion>/.
"""
from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.labels import EMOTIONS  # noqa: E402

CFG = load_config()
SAMPLE_RATE = int(CFG.audio.sample_rate)
DURATION_S = float(CFG.audio.duration_seconds)
OUT_ROOT = resolve_path(CFG.paths.data_raw_audio)


def _record_one(seconds: float) -> np.ndarray:
    import sounddevice as sd
    print(f"  ● Recording {seconds:.1f}s ...", end="", flush=True)
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    print(" done.")
    return audio.flatten()


def _save_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)         # int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())


def _prompt_int(message: str, default: int) -> int:
    raw = input(f"{message} [{default}]: ").strip()
    return int(raw) if raw else default


def main() -> int:
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        print("`sounddevice` is required. Install with:  pip install sounddevice")
        return 1

    print("=== Microphone fallback recorder ===")
    print(f"Sample rate: {SAMPLE_RATE} Hz, duration per take: {DURATION_S}s")
    takes = _prompt_int("Takes per emotion", 10)
    print(f"\nWe will record {takes} samples for each of: {', '.join(EMOTIONS)}")
    input("Press <Enter> when ready... ")

    for emotion in EMOTIONS:
        print(f"\n--- {emotion} ---")
        print(f"Speak with a {emotion.lower()} tone. {takes} takes coming up.")
        for i in range(takes):
            input(f"  [{i + 1}/{takes}] press <Enter> to start...")
            samples = _record_one(DURATION_S)
            ts = int(time.time() * 1000)
            out = OUT_ROOT / emotion / f"manual_{emotion.lower()}_{ts}.wav"
            _save_wav(out, samples)
            print(f"    saved → {out.relative_to(resolve_path('.'))}")

    print("\n✓ All recordings saved.")
    print("▶ NEXT: python src/data/validate_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
