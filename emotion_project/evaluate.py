"""Evaluate all three trained models on their held-out test sets.

For each available checkpoint, prints:
    * top-1 accuracy
    * macro F1
    * per-class precision/recall/F1
    * confusion matrix (text + a PNG saved to logs/)

Run:
    python evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.datasets import make_audio_loaders, make_face_loaders, make_fusion_loaders
from src.fusion.fusion_model import build_fusion_model
from src.models.audio_model import AudioNet
from src.models.face_model import FaceNet
from src.utils.config import load_config, resolve_path
from src.utils.helpers import load_checkpoint, select_device
from src.utils.labels import EMOTIONS


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _print_block(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    acc = (y_true == y_pred).mean()
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n=== {name} ===")
    print(f"  accuracy = {acc:.4f}   macro_f1 = {macro:.4f}")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTIONS))))
    print("  confusion matrix (rows=true, cols=pred):")
    width = max(len(e) for e in EMOTIONS)
    header = " " * (width + 2) + "  ".join(e[:6].rjust(6) for e in EMOTIONS)
    print(f"  {header}")
    for i, row in enumerate(cm):
        cells = "  ".join(f"{v:>6d}" for v in row)
        print(f"  {EMOTIONS[i].ljust(width)}  {cells}")
    _save_confusion_png(name, cm)


def _save_confusion_png(name: str, cm: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    out_dir = resolve_path("logs") / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"confusion_{name.lower()}.png"
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name} — confusion matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  saved {out_path.relative_to(resolve_path('.'))}")


# --------------------------------------------------------------------------- #
# Face                                                                        #
# --------------------------------------------------------------------------- #
def _eval_face(cfg, device) -> None:
    ckpt_path = resolve_path(cfg.paths.checkpoints) / "face_best.pt"
    if not ckpt_path.exists():
        print(f"  (skip face — no checkpoint at {ckpt_path})")
        return
    model = FaceNet(
        num_classes=int(cfg.project.num_classes),
        embedding_dim=int(cfg.face.embedding_dim),
        dropout=float(cfg.face.dropout),
        pretrained=False,
    ).to(device)
    model.load_state_dict(load_checkpoint(ckpt_path, device)["model_state"])
    model.eval()

    _, _, test_loader = make_face_loaders(batch_size=int(cfg.face.batch_size))
    preds, gts = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            out = model(imgs.to(device))
            preds.append(out.logits.argmax(1).cpu().numpy())
            gts.append(labels.numpy())
    _print_block("Face", np.concatenate(gts), np.concatenate(preds))


# --------------------------------------------------------------------------- #
# Audio                                                                       #
# --------------------------------------------------------------------------- #
def _eval_audio(cfg, device) -> None:
    ckpt_path = resolve_path(cfg.paths.checkpoints) / "audio_best.pt"
    if not ckpt_path.exists():
        print(f"  (skip audio — no checkpoint at {ckpt_path})")
        return
    state = load_checkpoint(ckpt_path, device)
    input_channels = int(state.get("input_channels") or 0)
    if input_channels <= 0:
        blob = torch.load(resolve_path(cfg.paths.audio_tensor), map_location="cpu", weights_only=False)
        input_channels = int(blob["channels"])

    model = AudioNet(
        input_channels=input_channels,
        num_classes=int(cfg.project.num_classes),
        hidden_dim=int(cfg.audio.lstm_hidden),
        num_layers=int(cfg.audio.lstm_layers),
        bidirectional=bool(cfg.audio.bidirectional),
        embedding_dim=int(cfg.audio.embedding_dim),
        dropout=float(cfg.audio.dropout),
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    _, _, test_loader = make_audio_loaders(batch_size=int(cfg.audio.batch_size))
    preds, gts = [], []
    with torch.no_grad():
        for feats, labels in test_loader:
            out = model(feats.to(device))
            preds.append(out.logits.argmax(1).cpu().numpy())
            gts.append(labels.numpy())
    _print_block("Audio", np.concatenate(gts), np.concatenate(preds))


# --------------------------------------------------------------------------- #
# Fusion                                                                      #
# --------------------------------------------------------------------------- #
def _eval_fusion(cfg, device) -> None:
    fusion_ckpt = resolve_path(cfg.paths.checkpoints) / "fusion_best.pt"
    if not fusion_ckpt.exists():
        print(f"  (skip fusion — no checkpoint at {fusion_ckpt})")
        return
    state = load_checkpoint(fusion_ckpt, device)

    face_net = FaceNet(
        num_classes=int(cfg.project.num_classes),
        embedding_dim=int(cfg.face.embedding_dim),
        dropout=float(cfg.face.dropout),
        pretrained=False,
    ).to(device)

    audio_blob = torch.load(resolve_path(cfg.paths.audio_tensor), map_location="cpu", weights_only=False)
    audio_net = AudioNet(
        input_channels=int(audio_blob["channels"]),
        num_classes=int(cfg.project.num_classes),
        hidden_dim=int(cfg.audio.lstm_hidden),
        num_layers=int(cfg.audio.lstm_layers),
        bidirectional=bool(cfg.audio.bidirectional),
        embedding_dim=int(cfg.audio.embedding_dim),
        dropout=float(cfg.audio.dropout),
    ).to(device)

    model = build_fusion_model(
        kind=str(state.get("fusion_type") or cfg.fusion.type),
        face_net=face_net,
        audio_net=audio_net,
        num_classes=int(cfg.project.num_classes),
        hidden_dim=int(cfg.fusion.hidden_dim),
        dropout=float(cfg.fusion.dropout),
        freeze_backbones=False,
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    _, _, test_loader = make_fusion_loaders(batch_size=int(cfg.fusion.batch_size))
    preds, gts = [], []
    face_w_sum, audio_w_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, feats, labels in test_loader:
            out = model(imgs.to(device), feats.to(device))
            preds.append(out.logits.argmax(1).cpu().numpy())
            gts.append(labels.numpy())
            face_w_sum += float(out.face_weight.sum().item())
            audio_w_sum += float(out.audio_weight.sum().item())
            n += labels.size(0)
    _print_block("Fusion", np.concatenate(gts), np.concatenate(preds))
    if n:
        print(f"  avg modality contribution → face={face_w_sum / n:.3f}  audio={audio_w_sum / n:.3f}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    cfg = load_config()
    device = select_device(cfg.project.device)
    print(f"=== evaluate.py — device={device} ===")
    _eval_face(cfg, device)
    _eval_audio(cfg, device)
    _eval_fusion(cfg, device)
    print("\n▶ NEXT: streamlit run app/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
