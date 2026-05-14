# Multimodal Emotion Recognition — Developer Guide

Production-quality PyTorch project that fuses facial expressions and speech
audio to classify five emotions: **Happy, Sad, Angry, Fearful, Neutral**.

This README is the **technical entry point** — install, run, train. For the
high-level project pitch see the [root README](../README.md). For the
phase-by-phase build plan see [`PLAN.md`](PLAN.md). For a complete narrative
explanation of every design decision (Arabic + English, ~700 lines) see
[`GUIDE.md`](GUIDE.md).

---

## Two ways to run inference

### 1. AI Mode (default in the app — easiest)

Uses two foundation models pulled from HuggingFace on first run (~1.5 GB
downloaded once into `~/.cache/huggingface`):

| Modality | Model |
|----------|-------|
| Face  | `dima806/facial_emotions_image_detection` (ViT-base, 7 → 5 emotions) |
| Audio | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (Wav2Vec2-large, 8 → 5 emotions) |

```bash
pip install -r requirements.txt
pip install praat-parselmouth transformers   # praat-parselmouth is also used
                                              # by the custom-mode pipeline
streamlit run app/app.py
```

### 2. Custom-trained models (full pipeline)

Train everything from scratch on RAVDESS + CREMA-D + TESS (audio) and FER2013
(face):

```bash
# 1. Get the data
python src/data/download_data.py        # RAVDESS + FER2013 base
python src/data/validate_data.py
# CREMA-D + TESS are pulled from HuggingFace mirrors (myleslinder/crema-d,
# myleslinder/tess) — see PLAN.md §Phase 7 for the exact commands.

# 2. Preprocess
python src/data/preprocess_faces.py
python src/data/preprocess_audio.py     # MFCC + mel + ZCR + prosodic (111 channels)
python src/data/build_dataset.py        # stratified 70/15/15 splits

# 3. Train
python train_face.py                    # ResNet-18, label smoothing, ~13 min on GPU
python train_audio.py                   # BiLSTM with prosodic features, ~10 min
python train_fusion.py                  # late fusion, ~5 min

# 4. Evaluate
python evaluate.py                      # accuracy / F1 / confusion matrices

# 5. Demo (AI mode by default — edit app/app.py to swap to custom models)
streamlit run app/app.py
```

---

## Layout

```
emotion_project/
├── PLAN.md                    # phase-by-phase plan + current status
├── GUIDE.md                   # full project guide (Arabic + English)
├── README.md                  # this file
├── requirements.txt
├── configs/
│   └── config.yaml            # ALL hyperparameters in one place
│
├── app/
│   └── app.py                 # Streamlit live demo (AI mode default)
│
├── data/
│   ├── raw/{audio,faces}/<Emotion>/
│   └── processed/
│       ├── faces.pt           # (31338, 3, 48, 48)
│       ├── audio.pt           # (9227, 111, 94)
│       └── splits/            # train/val/test manifests
│
├── checkpoints/               # face_best.pt, audio_best.pt, fusion_best.pt
├── logs/                      # TensorBoard runs
├── notebooks/                 # exploratory analysis
│
├── src/
│   ├── data/                  # download, preprocess, datasets
│   ├── models/                # FaceNet, AudioNet
│   ├── fusion/                # FusionNet (late + attention)
│   ├── ai/                    # foundation_models wrappers (Phase 8)
│   └── utils/
│       ├── labels.py          # canonical 5-emotion mapping
│       ├── audio_features.py  # shared MFCC + prosodic extractor
│       ├── config.py          # YAML loader
│       └── helpers.py         # device select, EarlyStopper, checkpoints
│
├── train_face.py
├── train_audio.py
├── train_fusion.py
└── evaluate.py
```

---

## Current dataset state

| Modality | Files | Source |
|----------|------:|--------|
| Audio | 9,227 | RAVDESS + CREMA-D + TESS |
| Faces | 31,338 | FER2013 (HuggingFace mirror) |

Audio split per class: Happy 1,863 · Sad 1,863 · Angry 1,863 · Fearful 1,863 · Neutral 1,775.

---

## Trained checkpoints (custom mode)

| Checkpoint | Test accuracy | Test macro-F1 |
|------------|--------------:|--------------:|
| `face_best.pt`   | 65.77% | 62.76% |
| `audio_best.pt`  | 74.01% | 73.93% |
| `fusion_best.pt` | 76.86% | 75.87% |

---

## Label mapping

| Index | Emotion |
|------:|---------|
| 0 | Happy |
| 1 | Sad |
| 2 | Angry |
| 3 | Fearful |
| 4 | Neutral |

Source of truth: [`src/utils/labels.py`](src/utils/labels.py). All datasets
are normalized via `canonicalize()` (e.g. RAVDESS code `05` → `"Angry"`,
CREMA-D `ANG` → `"Angry"`, TESS `angry` → `"Angry"`).

---

## Tips for live-demo accuracy

- Good lighting on the face.
- Sit close enough that the face occupies ≥1/3 of the frame.
- Speak normally — the model trained on natural speech, not acted speech.
- Quiet-ish background — the VAD threshold (sidebar) gates obvious silence,
  but loud HVAC fans can still bias results.

See [`GUIDE.md §8`](GUIDE.md) for full troubleshooting.

---

## Where to read next

| If you want to… | Open |
|---|---|
| Understand the project at 30,000 feet | [root README](../README.md) |
| See the build plan + current status | [`PLAN.md`](PLAN.md) |
| Understand every design decision and concept | [`GUIDE.md`](GUIDE.md) |
| Read the model code | [`src/models/`](src/models/) |
| Read the AI-mode wrappers | [`src/ai/foundation_models.py`](src/ai/foundation_models.py) |
| Tune hyperparameters | [`configs/config.yaml`](configs/config.yaml) |
