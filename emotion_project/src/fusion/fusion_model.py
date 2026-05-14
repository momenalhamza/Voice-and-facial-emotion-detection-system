"""Multimodal fusion classifiers.

Two variants share the same I/O contract — both take pre-trained FaceNet and
AudioNet modules and produce a 5-way emotion logit vector plus a pair of
modality contribution weights used for explainability in the Streamlit demo.

  LateFusion:       concatenate embeddings → MLP → logits.
                    Modality contribution = norm ratio of each branch.
  AttentionFusion:  small attention block learns per-sample weights over the
                    two embeddings and returns them directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.audio_model import AudioNet
from src.models.face_model import FaceNet


@dataclass
class FusionOutput:
    logits: torch.Tensor
    face_weight: torch.Tensor    # (B,)  in [0,1], sums with audio_weight to 1
    audio_weight: torch.Tensor   # (B,)


def _maybe_freeze(module: nn.Module, freeze: bool) -> None:
    if freeze:
        for p in module.parameters():
            p.requires_grad = False
        module.eval()


# --------------------------------------------------------------------------- #
# Late fusion                                                                 #
# --------------------------------------------------------------------------- #
class LateFusion(nn.Module):
    """Concatenate face + audio embeddings → MLP → logits.

    Modality contribution is computed post-hoc as the L2-norm ratio of each
    embedding *after* projection — it's a heuristic, not a learned weight.
    """

    def __init__(
        self,
        face_net: FaceNet,
        audio_net: AudioNet,
        num_classes: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.4,
        freeze_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.face_net = face_net
        self.audio_net = audio_net
        _maybe_freeze(self.face_net, freeze_backbones)
        _maybe_freeze(self.audio_net, freeze_backbones)

        in_dim = face_net.embedding_dim + audio_net.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, face_x: torch.Tensor, audio_x: torch.Tensor) -> FusionOutput:
        face_out = self.face_net(face_x)
        audio_out = self.audio_net(audio_x)

        fused = torch.cat([face_out.embedding, audio_out.embedding], dim=1)
        logits = self.head(fused)

        # Contribution heuristic: relative L2 magnitude
        f_norm = face_out.embedding.norm(dim=1)
        a_norm = audio_out.embedding.norm(dim=1)
        total = f_norm + a_norm + 1e-8
        return FusionOutput(
            logits=logits,
            face_weight=f_norm / total,
            audio_weight=a_norm / total,
        )


# --------------------------------------------------------------------------- #
# Attention fusion                                                            #
# --------------------------------------------------------------------------- #
class AttentionFusion(nn.Module):
    """Per-sample learned soft-attention over the two modality embeddings.

    The attention block sees a small joint summary of both modalities and
    emits two non-negative weights that sum to one. These weights are also
    returned to the caller for visualization.
    """

    def __init__(
        self,
        face_net: FaceNet,
        audio_net: AudioNet,
        num_classes: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.4,
        freeze_backbones: bool = False,
    ) -> None:
        super().__init__()
        assert face_net.embedding_dim == audio_net.embedding_dim, (
            "AttentionFusion expects matching embedding dimensions for face/audio"
        )
        self.face_net = face_net
        self.audio_net = audio_net
        _maybe_freeze(self.face_net, freeze_backbones)
        _maybe_freeze(self.audio_net, freeze_backbones)

        dim = face_net.embedding_dim
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, face_x: torch.Tensor, audio_x: torch.Tensor) -> FusionOutput:
        face_out = self.face_net(face_x)
        audio_out = self.audio_net(audio_x)

        joint = torch.cat([face_out.embedding, audio_out.embedding], dim=1)
        weights = torch.softmax(self.attn(joint), dim=1)   # (B, 2)
        face_w = weights[:, 0]
        audio_w = weights[:, 1]

        mixed = (
            face_out.embedding * face_w.unsqueeze(1)
            + audio_out.embedding * audio_w.unsqueeze(1)
        )
        logits = self.head(mixed)
        return FusionOutput(logits=logits, face_weight=face_w, audio_weight=audio_w)


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #
def build_fusion_model(
    kind: str,
    face_net: FaceNet,
    audio_net: AudioNet,
    num_classes: int,
    hidden_dim: int,
    dropout: float,
    freeze_backbones: bool,
) -> nn.Module:
    """Pick a fusion variant from a string. Used by train_fusion.py."""
    kind = kind.lower()
    if kind == "late":
        return LateFusion(face_net, audio_net, num_classes, hidden_dim, dropout, freeze_backbones)
    if kind == "attention":
        return AttentionFusion(face_net, audio_net, num_classes, hidden_dim, dropout, freeze_backbones)
    raise ValueError(f"Unknown fusion type: {kind!r}. Expected 'late' or 'attention'.")
