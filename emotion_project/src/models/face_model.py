"""FaceNet — facial-emotion classifier built on a pretrained ResNet-18 backbone.

Forward output is a `FaceOutput` named-tuple with both:
    embedding  — (B, embedding_dim) used by the fusion model
    logits     — (B, num_classes) used by the face-only classification head

This dual output makes the same module reusable for unimodal training and as
a frozen feature extractor inside the fusion network.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class FaceOutput:
    embedding: torch.Tensor
    logits: torch.Tensor


class FaceNet(nn.Module):
    """ResNet-18 backbone → projection head → classifier head.

    Args:
        num_classes:   number of emotion labels (5 for this project).
        embedding_dim: dimension of the bottleneck used by the fusion model.
        dropout:       dropout probability applied before the projection head.
        pretrained:    whether to load ImageNet weights for the backbone.
    """

    def __init__(
        self,
        num_classes: int = 5,
        embedding_dim: int = 256,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Load ResNet-18; gracefully degrade to random init if weights cache is offline.
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        except Exception:  # pragma: no cover — offline or torchvision misconfig
            backbone = models.resnet18(weights=None)

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> FaceOutput:
        """x: (B, 3, H, W) → FaceOutput(embedding, logits)."""
        features = self.backbone(x)
        embedding = self.projection(self.dropout(features))
        logits = self.classifier(embedding)
        return FaceOutput(embedding=embedding, logits=logits)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return the embedding only (used during fusion inference)."""
        self.eval()
        return self.forward(x).embedding
