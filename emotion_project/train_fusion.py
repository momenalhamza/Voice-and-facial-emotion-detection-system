"""Train the multimodal fusion classifier (FaceNet + AudioNet → fusion head).

Loads pretrained FaceNet and AudioNet checkpoints (produced by `train_face.py`
and `train_audio.py`), assembles the fusion model per `fusion.type` in the
YAML config, and trains the joint network.

Run:
    python train_fusion.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.datasets import make_fusion_loaders
from src.fusion.fusion_model import build_fusion_model
from src.models.audio_model import AudioNet
from src.models.face_model import FaceNet
from src.utils.config import load_config, resolve_path
from src.utils.helpers import (
    EarlyStopper,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    select_device,
)


def _load_face_backbone(cfg, device) -> FaceNet:
    model = FaceNet(
        num_classes=int(cfg.project.num_classes),
        embedding_dim=int(cfg.face.embedding_dim),
        dropout=float(cfg.face.dropout),
        pretrained=False,                 # we'll load our own weights
    ).to(device)
    ckpt_path = resolve_path(cfg.paths.checkpoints) / "face_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing face checkpoint: {ckpt_path}. Run `python train_face.py` first."
        )
    state = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    return model


def _load_audio_backbone(cfg, device) -> AudioNet:
    ckpt_path = resolve_path(cfg.paths.checkpoints) / "audio_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing audio checkpoint: {ckpt_path}. Run `python train_audio.py` first."
        )
    state = load_checkpoint(ckpt_path, map_location=device)
    input_channels = int(state.get("input_channels") or 0)
    if input_channels <= 0:
        # Fallback: read from the audio blob
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
    return model


def _epoch(model: nn.Module, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, feats, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            out = model(imgs, feats)
            loss = criterion(out.logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct += (out.logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.project.seed)
    device = select_device(cfg.project.device)
    print(f"=== train_fusion.py — device={device}  type={cfg.fusion.type} ===")

    face_net = _load_face_backbone(cfg, device)
    audio_net = _load_audio_backbone(cfg, device)
    print(f"  Loaded pretrained backbones (face/audio).")

    model = build_fusion_model(
        kind=str(cfg.fusion.type),
        face_net=face_net,
        audio_net=audio_net,
        num_classes=int(cfg.project.num_classes),
        hidden_dim=int(cfg.fusion.hidden_dim),
        dropout=float(cfg.fusion.dropout),
        freeze_backbones=bool(cfg.fusion.freeze_backbones),
    ).to(device)
    print(f"  Fusion params: {count_parameters(model):,}")

    train_loader, val_loader, _ = make_fusion_loaders(batch_size=int(cfg.fusion.batch_size))
    print(f"  train batches: {len(train_loader)}   val batches: {len(val_loader)}")

    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable, lr=float(cfg.fusion.lr), weight_decay=float(cfg.fusion.weight_decay))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    writer = SummaryWriter(log_dir=str(resolve_path(cfg.paths.logs) / "fusion"))
    stopper = EarlyStopper(patience=int(cfg.fusion.early_stopping_patience))

    best_val_acc = 0.0
    ckpt_dir = resolve_path(cfg.paths.checkpoints)

    for epoch in range(1, int(cfg.fusion.epochs) + 1):
        t0 = time.time()
        train_loss, train_acc = _epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = _epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_acc)
        dt = time.time() - t0

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "fusion_type": str(cfg.fusion.type),
                    "config": dict(cfg.fusion),
                },
                ckpt_dir, "fusion_best.pt",
            )
            marker = "  ✓ new best"

        print(f"  epoch {epoch:>3}/{cfg.fusion.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
              f"[{dt:.1f}s]{marker}")

        if stopper.step(val_acc):
            print(f"  ⏹ early stopping at epoch {epoch} (best val_acc={stopper.best:.4f})")
            break

    writer.close()
    print(f"\n✓ Fusion training done. Best val_acc={best_val_acc:.4f}")
    print(f"  checkpoint → {ckpt_dir / 'fusion_best.pt'}")
    print("▶ NEXT: python evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
