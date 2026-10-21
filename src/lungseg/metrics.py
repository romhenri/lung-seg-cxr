"""Métrica Dice e loss de treino."""

import segmentation_models_pytorch as smp
import torch
from torch import nn


def dice(pred_bin: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Dice por imagem e depois a média. Entradas binárias com shape (N, ...)."""
    p = pred_bin.reshape(pred_bin.shape[0], -1).float()
    t = target.reshape(target.shape[0], -1).float()
    inter = (p * t).sum(1)
    per_image = (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)
    return per_image.mean().item()


def combined_loss():
    """DiceLoss + BCEWithLogits, ambas sobre logits."""
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()
    return lambda logits, target: dice_loss(logits, target) + bce(logits, target)
