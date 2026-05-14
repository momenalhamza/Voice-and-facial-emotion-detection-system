"""AudioNet — speech-emotion classifier on top of a 2-layer LSTM.

Input shape: (B, T, C) where C = n_mfcc + n_mels + 1 (zero-crossing rate).

Forward output is an `AudioOutput` named-tuple with both:
    embedding  — (B, embedding_dim) for the fusion model
    logits     — (B, num_classes) for the audio-only classification head
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AudioOutput:
    embedding: torch.Tensor
    logits: torch.Tensor


class AudioNet(nn.Module):
    """Bidirectional LSTM over per-frame audio features.

    Args:
        input_channels: feature dimensionality per timestep
                        (= n_mfcc + n_mels + 1 for ZCR by default).
        num_classes:    number of emotion labels.
        hidden_dim:     LSTM hidden state size per direction.
        num_layers:     number of stacked LSTM layers.
        bidirectional:  whether to run the LSTM bidirectionally.
        embedding_dim:  bottleneck used by the fusion model.
        dropout:        applied inside the LSTM and before the projection head.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        embedding_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        directions = 2 if bidirectional else 1
        feat_dim = hidden_dim * directions

        # Attention pool over time so embedding doesn't depend on hand-picked timestep
        self.attn = nn.Linear(feat_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim

    def _attention_pool(self, hidden_seq: torch.Tensor) -> torch.Tensor:
        """Weighted average over the time axis.  hidden_seq: (B, T, F) → (B, F)."""
        scores = self.attn(hidden_seq).squeeze(-1)              # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)    # (B, T, 1)
        return (hidden_seq * weights).sum(dim=1)                # (B, F)

    def forward(self, x: torch.Tensor) -> AudioOutput:
        """x: (B, T, C) → AudioOutput(embedding, logits)."""
        hidden_seq, _ = self.lstm(x)            # (B, T, hidden*directions)
        pooled = self._attention_pool(hidden_seq)
        embedding = self.projection(self.dropout(pooled))
        logits = self.classifier(embedding)
        return AudioOutput(embedding=embedding, logits=logits)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x).embedding
