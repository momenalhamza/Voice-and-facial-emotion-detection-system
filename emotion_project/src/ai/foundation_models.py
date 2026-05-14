"""Pretrained foundation-model emotion classifiers (HuggingFace).

Replaces our custom-trained backbones with two strong off-the-shelf models:

    Face  → dima806/facial_emotions_image_detection (ViT)
    Audio → ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (Wav2Vec2)

Both expose a single helper that returns a probability vector over the
project's five canonical emotions (Happy, Sad, Angry, Fearful, Neutral).
Classes the source model has but we don't (disgust, surprise, calm) are
dropped or re-mapped, and the surviving probabilities are renormalized.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.utils.labels import EMOTIONS, LABEL_TO_INDEX

# Source label → canonical or None (drop)
_FACE_MAP: dict[str, str | None] = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "fear": "Fearful",
    "fearful": "Fearful",
    "neutral": "Neutral",
    "disgust": None,
    "surprise": None,
    "surprised": None,
}

_AUDIO_MAP: dict[str, str | None] = {
    "hap": "Happy", "happy": "Happy",
    "sad": "Sad",
    "ang": "Angry", "angry": "Angry",
    "fea": "Fearful", "fear": "Fearful", "fearful": "Fearful",
    "neu": "Neutral", "neutral": "Neutral",
    "calm": "Neutral",   # treat calm as neutral
    "dis": None, "disgust": None,
    "sur": None, "surprise": None, "surprised": None,
    "ps": None,          # pleasant surprise (TESS naming, just in case)
}

FACE_MODEL_ID = "dima806/facial_emotions_image_detection"
AUDIO_MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"


@dataclass
class _FaceBundle:
    processor: object
    model: torch.nn.Module
    id2label: dict[int, str]
    device: torch.device


@dataclass
class _AudioBundle:
    processor: object
    model: torch.nn.Module
    id2label: dict[int, str]
    sample_rate: int
    device: torch.device


def _remap_probs(src_probs: np.ndarray, id2label: dict[int, str], mapping: dict[str, str | None]) -> np.ndarray:
    """Aggregate source probabilities into the 5 canonical buckets, then renormalize."""
    out = np.zeros(len(EMOTIONS), dtype=np.float32)
    for i, raw in id2label.items():
        target = mapping.get(str(raw).lower())
        if target is None:
            continue
        out[LABEL_TO_INDEX[target]] += float(src_probs[i])
    s = out.sum()
    if s > 0:
        out /= s
    else:
        out[LABEL_TO_INDEX["Neutral"]] = 1.0  # safe default
    return out


def load_face_bundle(device: torch.device) -> _FaceBundle:
    from transformers import AutoModelForImageClassification, AutoProcessor
    processor = AutoProcessor.from_pretrained(FACE_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(FACE_MODEL_ID).half().to(device).eval()
    return _FaceBundle(processor=processor, model=model, id2label=model.config.id2label, device=device)


class _EhcalabresClassifierHead(torch.nn.Module):
    """Two-layer classifier head with tanh, matching the ehcalabres checkpoint
    layout (`classifier.dense` + `classifier.output`)."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.output = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.output(x)


class _Wav2Vec2EmotionClassifier(torch.nn.Module):
    """Wraps Wav2Vec2Model with the ehcalabres-style mean-pool classifier head."""

    def __init__(self, wav2vec2, hidden_size: int, num_labels: int):
        super().__init__()
        self.wav2vec2 = wav2vec2
        self.classifier = _EhcalabresClassifierHead(hidden_size, num_labels)

    @property
    def config(self):
        return self.wav2vec2.config

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
        out = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)
        return type("EmotionOut", (), {"logits": logits})()


def load_audio_bundle(device: torch.device) -> _AudioBundle:
    """Load ehcalabres wav2vec2 with its non-standard classifier head."""
    import glob
    from safetensors.torch import load_file
    from transformers import AutoFeatureExtractor, Wav2Vec2Config, Wav2Vec2Model

    processor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_ID)
    config = Wav2Vec2Config.from_pretrained(AUDIO_MODEL_ID)
    base = Wav2Vec2Model(config)
    model = _Wav2Vec2EmotionClassifier(
        base, hidden_size=int(config.hidden_size), num_labels=int(config.num_labels)
    )

    paths = glob.glob(
        f"{Path.home()}/.cache/huggingface/hub/models--{AUDIO_MODEL_ID.replace('/', '--')}/snapshots/*/model.safetensors"
    )
    if not paths:
        raise FileNotFoundError("Could not find cached ehcalabres weights — run `python -c \"from transformers import AutoModel; AutoModel.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')\"` first.")
    state = load_file(paths[0])
    missing, unexpected = model.load_state_dict(state, strict=False)
    classifier_missing = [k for k in missing if "classifier" in k]
    if classifier_missing:
        raise RuntimeError(f"Classifier weights missing after load: {classifier_missing}")

    model.half().to(device).eval()
    sr = int(getattr(processor, "sampling_rate", 16000))
    id2label = {int(k): v for k, v in config.id2label.items()}
    return _AudioBundle(processor=processor, model=model, id2label=id2label,
                        sample_rate=sr, device=device)


@torch.no_grad()
def face_predict(bundle: _FaceBundle, rgb_image: np.ndarray) -> np.ndarray:
    """rgb_image: (H, W, 3) uint8 RGB → 5-class probability vector."""
    inputs = bundle.processor(images=rgb_image, return_tensors="pt").to(bundle.device)
    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
    logits = bundle.model(**inputs).logits
    src = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return _remap_probs(src, bundle.id2label, _FACE_MAP)


@torch.no_grad()
def audio_predict(bundle: _AudioBundle, waveform: np.ndarray, source_sr: int) -> np.ndarray:
    """waveform: 1-D float32 mono, any sample rate (resampled here) → 5-class probs."""
    if source_sr != bundle.sample_rate:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=source_sr, target_sr=bundle.sample_rate)
    inputs = bundle.processor(
        waveform, sampling_rate=bundle.sample_rate, return_tensors="pt", padding=True
    )
    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
    inputs = {k: v.to(bundle.device) for k, v in inputs.items()}
    logits = bundle.model(**inputs).logits
    src = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return _remap_probs(src, bundle.id2label, _AUDIO_MAP)
