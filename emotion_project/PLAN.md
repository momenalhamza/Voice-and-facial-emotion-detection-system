# Multimodal Emotion Recognition — Project Plan

## Goal
Build a production-quality multimodal emotion recognition system that fuses
**facial expressions** and **speech audio** to classify 5 emotions:

**Happy | Sad | Angry | Fearful | Neutral**

The system supports two inference modes:
1. **Custom-trained mode** (Phases 0–6): face ResNet-18 + audio BiLSTM + late-fusion.
2. **AI mode** (Phase 8, current default in the app): foundation models
   (ViT for face, Wav2Vec2 for audio) with weighted-average fusion.

A Streamlit app provides a real-time webcam + microphone demo with both modes.

---

## Phase Checklist

- [x] **Phase 0 — Project skeleton**
  - Folder structure, `PLAN.md`, `README.md`, `requirements.txt`,
    `configs/config.yaml`, `__init__.py` package markers.

- [x] **Phase 1 — Data acquisition**  ✅ data on disk
  - `src/data/download_data.py` — RAVDESS (Zenodo) → CREMA-D (GitHub) → FER2013 (HuggingFace `Jeneral/fer-2013` pickle) → FER2013 (gdown) → Kaggle CLI.
  - `src/data/record_audio.py` — fallback microphone recorder.
  - `src/data/capture_faces.py` — fallback webcam capturer.
  - `src/data/cleanup_audio.py` — one-shot cleaner for the mis-labeled 07/08 RAVDESS wavs (run once after the labels fix).
  - `src/data/validate_data.py` — sanity-check what is on disk.

- [x] **Phase 2 — Preprocessing & feature extraction**  ✅ tensors on disk
  - `src/data/preprocess_faces.py` — MTCNN detect → 48×48 → normalize → `.pt`.
    Result: `data/processed/faces.pt` shape `(31338, 3, 48, 48)`.
  - `src/data/preprocess_audio.py` — MFCC + mel + ZCR + prosodic features → `.pt`.
  - `src/data/build_dataset.py` — stratified 70/15/15 splits.
  - `src/data/datasets.py` — `FaceDataset`, `AudioDataset`, `FusionDataset`.

- [x] **Phase 3 — Model architecture**
  - `src/models/face_model.py` — ResNet-18 backbone → 256-dim embedding + logits.
  - `src/models/audio_model.py` — 2-layer BiLSTM (hidden 256) → 256-dim embedding + logits.

- [x] **Phase 4 — Fusion**
  - `src/fusion/fusion_model.py` — late-fusion MLP + attention-fusion variant.

- [x] **Phase 5 — Training scripts**
  - `train_face.py` — face training with label smoothing + stronger augmentation (RandomErasing).
  - `train_audio.py`
  - `train_fusion.py`
  - TensorBoard logging, best-checkpoint saving, early stopping.

- [x] **Phase 6 — Evaluation & demo app (custom models)**
  - `evaluate.py` — accuracy / F1 / confusion matrix for all three models.
  - `app/app.py` (legacy live mode) — Streamlit webcam + mic demo with smoothing, VAD, modality-contribution bars.

- [x] **Phase 7 — Audio expansion + prosodic features**  ✅
  - Added **CREMA-D** (91 actors) and **TESS** (Toronto Emotional Speech Set) via HuggingFace mirrors.
  - Audio dataset grew from **1,056** → **9,227** wavs (×8.7) across ~117 speakers.
  - Added `src/utils/audio_features.py` — shared extractor used by both
    preprocessing and the app. New per-frame channels: log-F0, voicing
    probability, intensity (dB). New broadcast channels: jitter (local),
    shimmer (local), HNR mean — all via `praat-parselmouth`.
  - Feature dim: **105 → 111** channels.

- [x] **Phase 8 — Foundation-model AI mode**  ✅ (current app default)
  - `src/ai/foundation_models.py` — ViT (`dima806/facial_emotions_image_detection`)
    + Wav2Vec2 (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`).
  - Includes a custom `_Wav2Vec2EmotionClassifier` because the published
    checkpoint uses a non-standard classifier head (`classifier.dense` +
    `classifier.output`) that does not load via `AutoModelForAudioClassification`
    in transformers 5.x.
  - Source labels are remapped to the project's 5 canonical emotions
    (disgust/surprise dropped, calm → neutral, fear/fearful unified).
  - `app/app.py` rewritten to use these models with weighted-average fusion,
    EMA smoothing, VAD silence gating, and an "AI Mode" branded UI.

---

## Current Status

**Current counts**

| Modality | Files | Source |
|----------|-------|--------|
| Audio | 9,227 | RAVDESS + CREMA-D + TESS |
| Faces | 31,338 | FER2013 (HuggingFace mirror) |

**Audio split**: Happy 1,863 / Sad 1,863 / Angry 1,863 / Fearful 1,863 / Neutral 1,775.
**Face split**: Happy 8,989 / Sad 6,077 / Angry 4,953 / Fearful 5,121 / Neutral 6,198.

**Trained checkpoints (custom models)**:
- `checkpoints/face_best.pt` — ResNet-18, val_acc 67.24%, test_acc 65.77%.
- `checkpoints/audio_best.pt` — BiLSTM with prosodic features (111 ch), val_acc 72.90%, test_acc 74.01%.
- `checkpoints/fusion_best.pt` — late fusion, val_acc 78.07%, test_acc 76.86%.

**Default inference (AI mode)** uses foundation models from `~/.cache/huggingface`.

---

## How to Run Each Phase

### 0. Install dependencies
```
cd emotion_project
pip install -r requirements.txt
pip install praat-parselmouth transformers  # for Phases 7 & 8
```

### 1. Get the data
```
python src/data/download_data.py        # RAVDESS + FER2013 baseline
python src/data/validate_data.py
# Phase 7: CREMA-D + TESS are downloaded from HuggingFace mirrors
# (myleslinder/crema-d, myleslinder/tess) and ingested by `_ingest_audio_extras.py`
# style logic — see GUIDE.md.
```

### 2. Preprocess (custom models)
```
python src/data/preprocess_faces.py
python src/data/preprocess_audio.py       # now includes prosodic features
python src/data/build_dataset.py
```

### 3. Train (in this order)
```
python train_face.py
python train_audio.py
python train_fusion.py
```

### 4. Evaluate + demo
```
python evaluate.py                        # custom models
streamlit run app/app.py                  # AI-mode live demo
```

---

## Data Layout (after Phase 1+7)
```
data/raw/faces/<emotion>/*.jpg|png
data/raw/audio/<emotion>/*.wav        # RAVDESS + crema_*.wav + tess_*.wav
data/processed/
    faces.pt            # tensor of preprocessed face crops + labels
    audio.pt            # 9227 × 111 × 94 — features + labels
    splits/             # train/val/test manifests
```

## Label Mapping
| Index | Emotion  |
|-------|----------|
| 0     | Happy    |
| 1     | Sad      |
| 2     | Angry    |
| 3     | Fearful  |
| 4     | Neutral  |

Defined once in `src/utils/labels.py` and imported everywhere.

See **GUIDE.md** for a full, narrative explanation of the project for both
programmers and non-programmers.
