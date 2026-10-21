"""Avaliação no conjunto de validação: Dice e erro da razão D/E."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import CXRDataset, get_split, load_cache, subset, val_transforms
from .device import get_device
from .metrics import dice
from .model import load_checkpoint
from .postprocess import lung_stats
from .predict import render_overlay
from .train import predict_batches

log = logging.getLogger(__name__)


def evaluate(cfg: Config) -> None:
    device = get_device()
    model, ckpt = load_checkpoint(cfg.out / "best.pt", device)
    cache = load_cache(cfg)
    val_keys = get_split(cfg, cache["keys"])["val"]
    idx = subset(cache, val_keys)
    images, masks = cache["images"][idx], cache["masks"][idx]
    loader = DataLoader(CXRDataset(images, masks, val_transforms()), batch_size=16)
    preds = torch.cat([p for p, _ in predict_batches(model, loader, device, cfg.threshold)])
    preds = preds[:, 0].numpy().astype(np.uint8)

    rows, pred_stats = [], []
    for key, pred, gt in zip(val_keys, preds, masks, strict=True):
        ps, gs = lung_stats(pred), lung_stats(gt)
        clean = ps.clean_mask if ps else np.zeros_like(pred)
        pred_stats.append(ps)
        rows.append(
            {
                "key": key,
                "dice": dice(torch.from_numpy(pred[None]), torch.from_numpy(gt[None])),
                "dice_pos": dice(torch.from_numpy(clean[None]), torch.from_numpy(gt[None])),
                "razao_real": gs.razao if gs else np.nan,
                "razao_prevista": ps.razao if ps else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df["erro_razao"] = (df.razao_prevista - df.razao_real).abs()
    df.to_csv(cfg.out / "eval.csv", index=False)

    print(f"checkpoint: época {ckpt['epoch']}, dice val no treino {ckpt['val_dice']:.4f}")
    print(f"imagens de validação: {len(df)}")
    print(f"Dice médio (bruto):           {df.dice.mean():.4f}")
    print(f"Dice médio (pós-processado):  {df.dice_pos.mean():.4f}")
    print(
        f"Erro absoluto médio da razão: {df.erro_razao.mean():.4f}"
        f"  ({df.erro_razao.notna().sum()} imagens com dois pulmões)"
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df.razao_real, df.razao_prevista, s=12, alpha=0.7)
    both = df[["razao_real", "razao_prevista"]].to_numpy()
    lo, hi = np.nanmin(both), np.nanmax(both)
    ax.plot([lo, hi], [lo, hi], "r--", label="y = x")
    ax.set(xlabel="razão D/E real", ylabel="razão D/E prevista", title="Razão D/E (validação)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.out / "ratio_scatter.png", dpi=120)

    order = df.dice.sort_values().index
    picks = [("melhor", i) for i in order[::-1][:2]] + [("pior", i) for i in order[:2]]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (tag, i) in zip(axes, picks, strict=True):
        ax.imshow(render_overlay(images[i], pred_stats[i]))
        ax.set_title(f"{tag}: {df.key[i]}\nDice {df.dice[i]:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Azul = pulmão direito, verde = pulmão esquerdo")
    fig.tight_layout()
    fig.savefig(cfg.out / "best_worst.png", dpi=120)
    log.info("salvos eval.csv, ratio_scatter.png e best_worst.png em %s", cfg.out)
