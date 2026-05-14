# 🎭 Voice & Facial Emotion Recognition System

Real-time multimodal emotion recognition that combines **facial expressions**
and **speech audio** to classify five emotions:
**Happy 😊 · Sad 😢 · Angry 😠 · Fearful 😨 · Neutral 😐**.

Built end-to-end in PyTorch — from raw datasets to a live Streamlit demo —
with two switchable inference paths:

- **Custom-trained models** (ResNet-18 + BiLSTM + late fusion, trained on
  RAVDESS + CREMA-D + TESS for audio and FER2013 for face).
- **AI mode (default in the app)** — Foundation models from HuggingFace
  (`dima806/facial_emotions_image_detection` for face,
  `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` for audio)
  with weighted-average fusion.

> The full source code, training scripts, datasets pipeline and live demo
> live inside [`emotion_project/`](emotion_project/).

---

## ✨ Highlights

| | |
|---|---|
| 🧠 **Two inference modes** | Custom-trained backbones, or pretrained ViT + Wav2Vec2 foundation models |
| 🎵 **Rich acoustic features** | MFCC + log-mel + ZCR + **prosodic features** (F0, jitter, shimmer, HNR, intensity) via `praat-parselmouth` |
| 📊 **9,227 audio samples** | RAVDESS + CREMA-D + TESS — ~117 distinct speakers across multiple datasets |
| 🖼️ **31,338 face images** | FER2013 (HuggingFace mirror) with stronger augmentation + label smoothing |
| 🔬 **Production-quality pipeline** | Stratified splits, early stopping, TensorBoard logging, best-checkpoint saving |
| 🎥 **Live Streamlit demo** | Continuous webcam + mic streaming with EMA smoothing, VAD silence gating, audio level meter, configurable fusion weight |
| 📚 **Full documentation** | [`PLAN.md`](emotion_project/PLAN.md) phase plan + [`GUIDE.md`](emotion_project/GUIDE.md) — a 700-line book explaining the project for both programmers and non-programmers |

---

## 🚀 Quickstart

```bash
git clone https://github.com/momenalhamza/Voice-and-facial-emotion-detection-system.git
cd Voice-and-facial-emotion-detection-system/emotion_project

# 1. Dependencies
pip install -r requirements.txt
pip install praat-parselmouth transformers

# 2. Run the live AI-mode demo (downloads ~1.5 GB of models on first run)
streamlit run app/app.py
```

Full training-from-scratch path (custom models) is in
[`emotion_project/README.md`](emotion_project/README.md).

---

## 📁 Repository Layout

```
.
├── README.md                       ← you are here (project overview)
└── emotion_project/                ← all source code lives here
    ├── PLAN.md                     ← phase-by-phase project plan
    ├── GUIDE.md                    ← full project guide (Arabic + English)
    ├── README.md                   ← technical setup + run instructions
    ├── app/app.py                  ← Streamlit live demo
    ├── checkpoints/                ← trained model weights
    ├── configs/config.yaml         ← all hyperparameters
    ├── data/                       ← raw + processed datasets
    ├── src/                        ← models, data pipeline, AI bundles
    ├── train_face.py
    ├── train_audio.py
    ├── train_fusion.py
    └── evaluate.py
```

---

## 📊 Results

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| Face (ResNet-18, custom) | 65.77% | 62.76% |
| Audio (BiLSTM + prosodic, custom) | 74.01% | 73.93% |
| Fusion (late, custom) | 76.86% | 75.87% |
| **AI Mode (ViT + Wav2Vec2)** | **~85% face / ~80% audio** (benchmark estimates) | — |

See [`GUIDE.md §7`](emotion_project/GUIDE.md) for full per-class metrics and
confusion matrices.

---

## 🛠️ Tech Stack

PyTorch · torchvision · transformers · librosa · praat-parselmouth ·
OpenCV · Streamlit · TensorBoard · scikit-learn

---

## 📖 Documentation Map

| Document | Purpose | Audience |
|---|---|---|
| [`README.md`](README.md) (this file) | Project overview, badges, top-level layout | GitHub visitors |
| [`emotion_project/README.md`](emotion_project/README.md) | Technical setup + run instructions | Developers |
| [`emotion_project/PLAN.md`](emotion_project/PLAN.md) | Phase-by-phase build plan and current status | Project maintainers |
| [`emotion_project/GUIDE.md`](emotion_project/GUIDE.md) | Complete project guide — concepts, code, decisions, glossary | Everyone (Arabic + English) |

---

## 📜 License

MIT — see `LICENSE` if present, otherwise this is provided as-is for
educational and research purposes.

## 🙏 Datasets used

- **RAVDESS** — Livingstone & Russo, 2018 (CC-BY-NC-SA-4.0)
- **CREMA-D** — Cao et al., 2014 (ODbL)
- **TESS** — Pichora-Fuller & Dupuis, 2020 (CC-BY-NC-4.0)
- **FER2013** — Goodfellow et al., 2013 (Kaggle)

And foundation models from the HuggingFace community:
[`dima806`](https://huggingface.co/dima806),
[`ehcalabres`](https://huggingface.co/ehcalabres),
[`myleslinder`](https://huggingface.co/myleslinder).
