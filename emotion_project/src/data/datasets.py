"""Dataset and DataLoader classes for both modalities.

Each `*Dataset` reads the corresponding `.pt` blob and a JSON split manifest,
then exposes `__getitem__(i) -> (input_tensor, label)`. Optional torchvision
transforms are wired in for face augmentation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.config import load_config, resolve_path

_CFG = load_config()


# --------------------------------------------------------------------------- #
# Face dataset                                                                #
# --------------------------------------------------------------------------- #
class FaceDataset(Dataset):
    """Reads preprocessed face crops + applies optional augmentations.

    Returns tuples of (image, label) where image is a float32 tensor shaped
    (3, H, W), normalized by the ImageNet stats configured in YAML.
    """

    def __init__(self, split: str, augment: bool = False) -> None:
        blob_path = resolve_path(_CFG.paths.faces_tensor)
        blob = torch.load(blob_path, map_location="cpu", weights_only=False)
        self.images: torch.Tensor = blob["images"]            # (N,3,H,W) float in [0,1]
        self.all_labels: torch.Tensor = blob["labels"]

        manifest = resolve_path(_CFG.paths.splits_dir) / f"face_{split}.json"
        with manifest.open() as fh:
            spec = json.load(fh)
        self.indices = spec["indices"]

        norm = transforms.Normalize(mean=_CFG.face.mean, std=_CFG.face.std)
        if augment:
            # Stronger augmentation reduces overfitting on the noisy FER2013 labels.
            self.transform: Callable = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)),
                norm,
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            ])
        else:
            self.transform = norm

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[i]
        img = self.images[idx]
        img = self.transform(img)
        return img, self.all_labels[idx]


# --------------------------------------------------------------------------- #
# Audio dataset                                                               #
# --------------------------------------------------------------------------- #
class AudioDataset(Dataset):
    """Reads preprocessed audio features.

    Returns (features, label) where features is shaped (T, C) — we transpose
    here so the LSTM sees `(batch, time, channels)` directly.
    """

    def __init__(self, split: str) -> None:
        blob_path = resolve_path(_CFG.paths.audio_tensor)
        blob = torch.load(blob_path, map_location="cpu", weights_only=False)
        self.features: torch.Tensor = blob["features"]        # (N, C, T)
        self.all_labels: torch.Tensor = blob["labels"]

        manifest = resolve_path(_CFG.paths.splits_dir) / f"audio_{split}.json"
        with manifest.open() as fh:
            spec = json.load(fh)
        self.indices = spec["indices"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[i]
        feat = self.features[idx]              # (C, T)
        return feat.transpose(0, 1).contiguous(), self.all_labels[idx]


# --------------------------------------------------------------------------- #
# Fusion dataset                                                              #
# --------------------------------------------------------------------------- #
class FusionDataset(Dataset):
    """Pairs samples across modalities for late-fusion training.

    Because public face and audio datasets aren't truly paired, we randomly
    pair samples that share the same emotion label. The pairing is fixed by a
    deterministic RNG seeded with the split name so each epoch sees the same
    pairings (important for fair val/test evaluation).
    """

    def __init__(self, split: str, augment_face: bool = False) -> None:
        self.face = FaceDataset(split=split, augment=augment_face)
        self.audio = AudioDataset(split=split)

        # Build label → audio-index buckets for the chosen split
        audio_labels = [int(l) for l in self.audio.all_labels[self.audio.indices]]
        self.audio_by_label: dict[int, list[int]] = {}
        for local_i, lbl in enumerate(audio_labels):
            self.audio_by_label.setdefault(lbl, []).append(local_i)

        face_labels = [int(l) for l in self.face.all_labels[self.face.indices]]
        rng = torch.Generator().manual_seed(hash(split) % (2**31))
        self.pairs: list[tuple[int, int]] = []
        for face_local_i, lbl in enumerate(face_labels):
            bucket = self.audio_by_label.get(lbl)
            if not bucket:
                # No matching audio sample in this split — skip
                continue
            j = int(torch.randint(0, len(bucket), (1,), generator=rng).item())
            self.pairs.append((face_local_i, bucket[j]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        face_i, audio_i = self.pairs[i]
        img, label = self.face[face_i]
        feat, _ = self.audio[audio_i]
        return img, feat, label


# --------------------------------------------------------------------------- #
# DataLoader builders                                                         #
# --------------------------------------------------------------------------- #
def make_face_loaders(batch_size: int, num_workers: int = 2):
    train = DataLoader(FaceDataset("train", augment=True),
                       batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val = DataLoader(FaceDataset("val"), batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=True)
    test = DataLoader(FaceDataset("test"), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
    return train, val, test


def make_audio_loaders(batch_size: int, num_workers: int = 2):
    train = DataLoader(AudioDataset("train"), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True)
    val = DataLoader(AudioDataset("val"), batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=True)
    test = DataLoader(AudioDataset("test"), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
    return train, val, test


def make_fusion_loaders(batch_size: int, num_workers: int = 2):
    train = DataLoader(FusionDataset("train", augment_face=True),
                       batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val = DataLoader(FusionDataset("val"), batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=True)
    test = DataLoader(FusionDataset("test"), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
    return train, val, test
