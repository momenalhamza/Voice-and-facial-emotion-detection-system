"""Train the face-only emotion classifier (FaceNet).

Reads hyperparameters from configs/config.yaml.
Logs scalars to logs/face/ (TensorBoard).
Writes best checkpoint to checkpoints/face_best.pt.

Run:
    python train_face.py
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
from src.data.datasets import make_face_loaders
from src.models.face_model import FaceNet
from src.utils.config import load_config, resolve_path
from src.utils.helpers import EarlyStopper, count_parameters, save_checkpoint, seed_everything, select_device


def _epoch(model: nn.Module, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            out = model(imgs)
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
    print(f"=== train_face.py — device={device} ===")

    train_loader, val_loader, _ = make_face_loaders(batch_size=int(cfg.face.batch_size))
    print(f"  train batches: {len(train_loader)}   val batches: {len(val_loader)}")

    model = FaceNet(
        num_classes=int(cfg.project.num_classes),
        embedding_dim=int(cfg.face.embedding_dim),
        dropout=float(cfg.face.dropout),
        pretrained=True,
    ).to(device)
    print(f"  FaceNet params: {count_parameters(model):,}")

    # Label smoothing helps with the well-known FER2013 label noise (~10%).
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr=float(cfg.face.lr), weight_decay=float(cfg.face.weight_decay))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    writer = SummaryWriter(log_dir=str(resolve_path(cfg.paths.logs) / "face"))
    stopper = EarlyStopper(patience=int(cfg.face.early_stopping_patience))

    best_val_acc = 0.0
    ckpt_dir = resolve_path(cfg.paths.checkpoints)

    for epoch in range(1, int(cfg.face.epochs) + 1):
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
                 "val_acc": val_acc, "config": dict(cfg.face)},
                ckpt_dir, "face_best.pt",
            )
            marker = "  ✓ new best"

        print(f"  epoch {epoch:>3}/{cfg.face.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
              f"[{dt:.1f}s]{marker}")

        if stopper.step(val_acc):
            print(f"  ⏹ early stopping at epoch {epoch} (best val_acc={stopper.best:.4f})")
            break

    writer.close()
    print(f"\n✓ Face training done. Best val_acc={best_val_acc:.4f}")
    print(f"  checkpoint → {ckpt_dir / 'face_best.pt'}")
    print("▶ NEXT: python train_audio.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
