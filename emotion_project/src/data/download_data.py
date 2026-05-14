"""Phase 1 — automated dataset download.

Strategy (audio):
    1. RAVDESS speech from Zenodo (direct HTTPS, no credentials needed).
    2. CREMA-D from the public GitHub mirror.
    3. If both fail → print precise manual instructions and exit.

Strategy (faces):
    1. FER2013 via Kaggle API if available.
    2. FER2013 via a public Google-Drive mirror through `gdown`.
    3. If both fail → print precise manual instructions and exit.

After this script runs, raw files are organized as:

    data/raw/audio/<Emotion>/*.wav
    data/raw/faces/<Emotion>/*.jpg

Files belonging to emotions outside the 5-class set are skipped.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import io
import pickle

import cv2
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# Make `src` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.labels import EMOTIONS, canonicalize  # noqa: E402

CFG = load_config()
RAW_AUDIO_DIR = resolve_path(CFG.paths.data_raw_audio)
RAW_FACES_DIR = resolve_path(CFG.paths.data_raw_faces)

# --- Public dataset endpoints (kept as constants so they're easy to update) ---
RAVDESS_ZENODO_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
CREMA_D_GITHUB_TARBALL = "https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.tar.gz"
FER2013_GDRIVE_ID = "1ZAfwZmU_5R4F2Hd9SuJP4Vd5MGgWXjib"  # community mirror; subject to availability

# HuggingFace `Jeneral/fer-2013` ships two pickle files (`train.pt`, `test.pt`)
# each containing a list of {"img_bytes": <bytes>, "labels": <int>} dicts.
# The repo's class order is angry/disgust/fear/happy/neutral/sad/surprise —
# different from canonical FER2013 (which puts neutral last).
FER2013_HF_URLS = {
    "train": "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main/train.pt",
    "test":  "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main/test.pt",
}
FER2013_HF_INDEX_TO_LABEL: dict[int, str | None] = {
    0: "Angry",
    1: None,            # disgust
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: None,            # surprise
}

# The pickle stores `labels` as strings rather than ints; map them directly.
FER2013_HF_NAME_TO_LABEL: dict[str, str | None] = {
    "angry": "Angry",
    "disgust": None,
    "fear": "Fearful",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": None,
}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _ensure_dirs() -> None:
    for emotion in EMOTIONS:
        (RAW_AUDIO_DIR / emotion).mkdir(parents=True, exist_ok=True)
        (RAW_FACES_DIR / emotion).mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path, desc: str) -> bool:
    """Stream `url` to `dest`. Returns True on success, False on any failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as fh, tqdm(
                desc=desc, total=total, unit="B", unit_scale=True, leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    fh.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as exc:  # noqa: BLE001 — network failures are varied
        print(f"  ✗ download failed ({desc}): {exc}")
        if dest.exists():
            dest.unlink()
        return False


def _safe(fn: Callable[[], bool], label: str) -> bool:
    print(f"\n>>> Attempting: {label}")
    try:
        ok = fn()
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ {label} raised {type(exc).__name__}: {exc}")
        return False
    print(f"  {'✓' if ok else '✗'} {label}: {'success' if ok else 'failed'}")
    return ok


# --------------------------------------------------------------------------- #
# RAVDESS                                                                     #
# --------------------------------------------------------------------------- #
def _try_ravdess() -> bool:
    """Download RAVDESS speech and organize by emotion."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zip_path = tmp / "ravdess.zip"
        if not _download(RAVDESS_ZENODO_URL, zip_path, "RAVDESS"):
            return False
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
        except zipfile.BadZipFile:
            return False

        n_kept = 0
        for wav in tmp.rglob("*.wav"):
            # RAVDESS filename: 03-01-{emotion:02d}-*.wav
            parts = wav.stem.split("-")
            if len(parts) < 3:
                continue
            label = canonicalize(parts[2])
            if label is None:
                continue
            shutil.copy(wav, RAW_AUDIO_DIR / label / wav.name)
            n_kept += 1
        print(f"  RAVDESS: copied {n_kept} wavs")
        return n_kept > 0


# --------------------------------------------------------------------------- #
# CREMA-D                                                                     #
# --------------------------------------------------------------------------- #
def _try_crema_d() -> bool:
    """Fallback audio source if RAVDESS is unavailable."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tar_path = tmp / "crema.tar.gz"
        if not _download(CREMA_D_GITHUB_TARBALL, tar_path, "CREMA-D"):
            return False
        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(tmp)
        except tarfile.TarError:
            return False

        n_kept = 0
        for wav in tmp.rglob("*.wav"):
            # CREMA-D filename: 1001_DFA_ANG_XX.wav
            parts = wav.stem.split("_")
            if len(parts) < 3:
                continue
            label = canonicalize(parts[2])
            if label is None:
                continue
            shutil.copy(wav, RAW_AUDIO_DIR / label / wav.name)
            n_kept += 1
        print(f"  CREMA-D: copied {n_kept} wavs")
        return n_kept > 0


# --------------------------------------------------------------------------- #
# FER2013                                                                     #
# --------------------------------------------------------------------------- #
def _try_fer2013_kaggle() -> bool:
    """Use Kaggle API if user has it configured (~/.kaggle/kaggle.json)."""
    if shutil.which("kaggle") is None:
        print("  (kaggle CLI not on PATH — `pip install kaggle` to enable)")
        return False
    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        print(f"  (no Kaggle credentials at {creds} — see https://www.kaggle.com/docs/api)")
        return False
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "msambare/fer2013", "-p", tmp, "--unzip"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stderr.strip())
            return False
        return _ingest_fer_directory(Path(tmp))


def _try_fer2013_huggingface() -> bool:
    """Download FER2013 from `Jeneral/fer-2013` on the HuggingFace Hub.

    The repo ships two pickle files (`train.pt`, `test.pt`), each a list of
    {"img_bytes": <bytes>, "labels": <int>}. We decode the image bytes with
    PIL and write JPGs into the matching emotion folder.

    pickle.load is used because that is the format the dataset is published
    in. The bytes come from a public, well-known HF Hub URL — the same path
    used by the official `datasets` library — and we never load anything
    callable from these payloads (we only consume `bytes` and `int`).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        total_kept = 0
        total_dropped = 0
        for split_name, url in FER2013_HF_URLS.items():
            pt_path = tmp / f"{split_name}.pt"
            if not _download(url, pt_path, f"FER2013 ({split_name})"):
                return False
            kept, dropped = _ingest_fer_huggingface_pickle(pt_path, split_name)
            total_kept += kept
            total_dropped += dropped
        print(f"  FER2013 (HF): wrote {total_kept} JPGs  "
              f"(dropped {total_dropped} disgust/surprise)")
        return total_kept > 0


def _ingest_fer_huggingface_pickle(pt_path: Path, split_name: str) -> tuple[int, int]:
    """Decode one HuggingFace FER2013 pickle file into per-emotion JPGs."""
    try:
        with pt_path.open("rb") as fh:
            examples = pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ pickle read failed for {split_name}: {exc}")
        return 0, 0

    kept = 0
    dropped = 0
    for i, ex in enumerate(tqdm(examples, desc=f"FER2013 {split_name}", leave=False)):
        try:
            raw = ex["img_bytes"]
            raw_label = ex["labels"]
        except (TypeError, KeyError):
            continue

        if isinstance(raw_label, str):
            label = FER2013_HF_NAME_TO_LABEL.get(raw_label.lower())
        else:
            try:
                label = FER2013_HF_INDEX_TO_LABEL.get(int(raw_label))
            except (TypeError, ValueError):
                label = None
        if label is None:
            dropped += 1
            continue
        try:
            img = Image.open(io.BytesIO(raw)).convert("L")
        except Exception:  # noqa: BLE001
            continue
        arr = np.array(img, dtype=np.uint8)
        out = RAW_FACES_DIR / label / f"fer2013_{split_name}_{i:05d}.jpg"
        cv2.imwrite(str(out), arr)
        kept += 1
    return kept, dropped


def _try_fer2013_gdrive() -> bool:
    """Public mirror via gdown."""
    try:
        import gdown
    except ImportError:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out = tmp / "fer2013.zip"
        try:
            gdown.download(id=FER2013_GDRIVE_ID, output=str(out), quiet=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ gdown failed: {exc}")
            return False
        if not out.exists():
            return False
        try:
            with zipfile.ZipFile(out) as zf:
                zf.extractall(tmp)
        except zipfile.BadZipFile:
            return False
        return _ingest_fer_directory(tmp)


def _ingest_fer_directory(root: Path) -> bool:
    """FER2013 mirrors usually have <split>/<emotion>/<idx>.jpg layout."""
    n_kept = 0
    for img in root.rglob("*.jpg"):
        # Walk parent folders looking for an emotion-named directory
        label = None
        for parent in img.parents:
            label = canonicalize(parent.name)
            if label is not None:
                break
        if label is None:
            continue
        shutil.copy(img, RAW_FACES_DIR / label / f"{img.parent.name}_{img.name}")
        n_kept += 1
    for img in root.rglob("*.png"):
        label = None
        for parent in img.parents:
            label = canonicalize(parent.name)
            if label is not None:
                break
        if label is None:
            continue
        shutil.copy(img, RAW_FACES_DIR / label / f"{img.parent.name}_{img.name}")
        n_kept += 1
    print(f"  FER2013: copied {n_kept} images")
    return n_kept > 0


# --------------------------------------------------------------------------- #
# Manual fallback messages                                                    #
# --------------------------------------------------------------------------- #
AUDIO_MANUAL = """\
Could not auto-download an audio dataset. Do ONE of the following:

  (A) Manually download RAVDESS speech:
      https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
      Unzip it anywhere, then drop the *.wav files into:
          data/raw/audio/<Emotion>/    (one of: Happy, Sad, Angry, Fearful, Neutral)
      The third number in each RAVDESS filename is the emotion code
      (03=Happy, 04=Sad, 05=Angry, 06=Fearful, 01/02=Neutral). Skip 07 and 08.

  (B) Record your own samples:
      python src/data/record_audio.py
      (minimum 10 samples per emotion recommended)

After your audio is in place, run:
    python src/data/validate_data.py
"""

FACES_MANUAL = """\
Could not auto-download a face dataset. Pick ONE option:

  (A) Set up the Kaggle API and re-run this script (recommended):
      1) pip install kaggle
      2) go to https://www.kaggle.com/<your-user>/account → "Create New API Token"
      3) place the downloaded kaggle.json at ~/.kaggle/kaggle.json
         (chmod 600 ~/.kaggle/kaggle.json)
      4) python src/data/download_data.py

  (B) Manually download FER2013 and drop in the files:
      https://www.kaggle.com/datasets/msambare/fer2013
      Unzip, then move .jpg files into:
          data/raw/faces/<Emotion>/    (Happy | Sad | Angry | Fearful | Neutral)
      The folders 'disgust' and 'surprise' should be skipped.

  (C) Capture your own samples from webcam (works offline, ~5 minutes):
      python src/data/capture_faces.py
      (minimum 50 images per emotion recommended)

After your images are in place, run:
    python src/data/validate_data.py
"""


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=== Phase 1 — Data acquisition ===")
    _ensure_dirs()

    audio_ok = _safe(_try_ravdess, "RAVDESS from Zenodo")
    if not audio_ok:
        audio_ok = _safe(_try_crema_d, "CREMA-D from GitHub")

    faces_ok = _safe(_try_fer2013_kaggle, "FER2013 via Kaggle API")
    if not faces_ok:
        faces_ok = _safe(_try_fer2013_huggingface, "FER2013 via HuggingFace (Jeneral/fer-2013)")
    if not faces_ok:
        faces_ok = _safe(_try_fer2013_gdrive, "FER2013 via gdown mirror")

    print("\n=== Summary ===")
    if audio_ok:
        print("  ✓ Audio dataset present in data/raw/audio/")
    else:
        print("  ✗ Audio dataset NOT downloaded")
        print(AUDIO_MANUAL)

    if faces_ok:
        print("  ✓ Face dataset present in data/raw/faces/")
    else:
        print("  ✗ Face dataset NOT downloaded")
        print(FACES_MANUAL)

    if audio_ok and faces_ok:
        print("\n▶ NEXT: python src/data/validate_data.py")
        return 0
    print("\n▶ NEXT: follow the manual steps above, then run "
          "`python src/data/validate_data.py`")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
