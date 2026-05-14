"""Train the audio-only emotion classifier (AudioNet).

Run:
    python train_audio.py
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
from src.data.datasets import make_audio_loaders
from src.models.audio_model import AudioNet
from src.utils.config import load_config, resolve_path
from src.utils.helpers import EarlyStopper, count_parameters, save_checkpoint, seed_everything, select_device


def _epoch(model: nn.Module, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for feats, labels in loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            out = model(feats)
            loss = criterion(out.logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * feats.size(0)
            correct += (out.logits.argmax(1) == labels).sum().item()
            total += feats.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


def _peek_input_channels(cfg) -> int:
    """Read C from the saved audio blob — no need to recompute."""
    blob = torch.load(resolve_path(cfg.paths.audio_tensor), map_location="cpu", weights_only=False)
    return int(blob["channels"])


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.project.seed)
    device = select_device(cfg.project.device)
    print(f"=== train_audio.py — device={device} ===")

    input_channels = _peek_input_channels(cfg)
    print(f"  input feature channels: {input_channels}")

    train_loader, val_loader, _ = make_audio_loaders(batch_size=int(cfg.audio.batch_size))
    print(f"  train batches: {len(train_loader)}   val batches: {len(val_loader)}")

    model = AudioNet(
        input_channels=input_channels,
        num_classes=int(cfg.project.num_classes),
        hidden_dim=int(cfg.audio.lstm_hidden),
        num_layers=int(cfg.audio.lstm_layers),
        bidirectional=bool(cfg.audio.bidirectional),
        embedding_dim=int(cfg.audio.embedding_dim),
        dropout=float(cfg.audio.dropout),
    ).to(device)
    print(f"  AudioNet params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=float(cfg.audio.lr), weight_decay=float(cfg.audio.weight_decay))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    writer = SummaryWriter(log_dir=str(resolve_path(cfg.paths.logs) / "audio"))
    stopper = EarlyStopper(patience=int(cfg.audio.early_stopping_patience))

    best_val_acc = 0.0
    ckpt_dir = resolve_path(cfg.paths.checkpoints)

    for epoch in range(1, int(cfg.audio.epochs) + 1):
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
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_acc": val_acc, "config": dict(cfg.audio),
                 "input_channels": input_channels},
                ckpt_dir, "audio_best.pt",
            )
            marker = "  ✓ new best"

        print(f"  epoch {epoch:>3}/{cfg.audio.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
              f"[{dt:.1f}s]{marker}")

        if stopper.step(val_acc):
            print(f"  ⏹ early stopping at epoch {epoch} (best val_acc={stopper.best:.4f})")
            break

    writer.close()
    print(f"\n✓ Audio training done. Best val_acc={best_val_acc:.4f}")
    print(f"  checkpoint → {ckpt_dir / 'audio_best.pt'}")
    print("▶ NEXT: python train_fusion.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
