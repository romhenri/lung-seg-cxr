"""Treino da U-Net."""

import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import CXRDataset, get_split, load_cache, subset, train_transforms, val_transforms
from .device import get_device
from .metrics import combined_loss, dice
from .model import build_model, save_checkpoint

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_batches(model, loader, device, threshold: float):
    """Gera (pred_bin, target) por batch, ambos na CPU."""
    model.eval()
    for x, y in loader:
        prob = torch.sigmoid(model(x.to(device))).cpu()
        yield prob > threshold, y


def train(cfg: Config) -> None:
    seed_everything(cfg.seed)
    device = get_device()
    cache = load_cache(cfg)
    split = get_split(cfg, cache["keys"])
    tr, va = subset(cache, split["train"]), subset(cache, split["val"])
    tc = cfg.train
    train_dl = DataLoader(
        CXRDataset(cache["images"][tr], cache["masks"][tr], train_transforms()),
        batch_size=tc.batch_size,
        shuffle=True,
        num_workers=tc.num_workers,
        drop_last=True,
    )
    val_dl = DataLoader(
        CXRDataset(cache["images"][va], cache["masks"][va], val_transforms()),
        batch_size=tc.batch_size,
        num_workers=tc.num_workers,
    )

    model = build_model(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.epochs)
    loss_fn = combined_loss()

    history, best = [], -1.0
    for epoch in range(1, tc.epochs + 1):
        model.train()
        losses = []
        bar = tqdm(train_dl, desc=f"época {epoch}/{tc.epochs}", leave=False)
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()

        batches = predict_batches(model, val_dl, device, cfg.threshold)
        val_dice = sum(dice(p, y) * len(y) for p, y in batches) / len(va)
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_dice": val_dice})
        improved = val_dice > best
        if improved:
            best = val_dice
            save_checkpoint(cfg.out / "best.pt", model, cfg, epoch, val_dice)
        log.info(
            "época %d: loss %.4f | dice val %.4f%s",
            epoch,
            train_loss,
            val_dice,
            "  (melhor, salvo)" if improved else "",
        )
        pd.DataFrame(history).to_csv(cfg.out / "train_log.csv", index=False)

    df = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    df.plot(x="epoch", y="train_loss", ax=ax[0], marker="o", title="Loss de treino")
    df.plot(x="epoch", y="val_dice", ax=ax[1], marker="o", title="Dice de validação")
    fig.tight_layout()
    fig.savefig(cfg.out / "curves.png", dpi=120)
    log.info("melhor Dice de validação: %.4f -> %s", best, cfg.out / "best.pt")
