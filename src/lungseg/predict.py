"""Inferência em uma imagem de qualquer tamanho e overlay colorido."""

from dataclasses import replace

import cv2
import numpy as np
import torch

from .data import val_transforms
from .postprocess import LungStats, lung_stats

COLOR_DIREITO = (0, 0, 255)  # azul (RGB)
COLOR_ESQUERDO = (0, 200, 0)  # verde (RGB)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    code = cv2.COLOR_RGBA2GRAY if img.shape[2] == 4 else cv2.COLOR_RGB2GRAY
    return cv2.cvtColor(img.astype(np.uint8), code)


@torch.no_grad()
def predict_image(
    model, img: np.ndarray, size: int, threshold: float, device: torch.device
) -> LungStats | None:
    """Segmenta uma imagem (RGB ou cinza). Os campos de LungStats vêm no tamanho original."""
    gray = to_gray(img)
    h, w = gray.shape
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    x = val_transforms()(image=np.repeat(small[..., None], 3, axis=2))["image"]
    prob = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
    stats = lung_stats((prob > threshold).astype(np.uint8))
    if stats is None:
        return None

    labels = cv2.resize(stats.labels, (w, h), interpolation=cv2.INTER_NEAREST)
    area_r, area_l = int((labels == 1).sum()), int((labels == 2).sum())
    sx, sy = w / size, h / size
    return replace(
        stats,
        area_direito=area_r,
        area_esquerdo=area_l,
        razao=area_r / max(area_l, 1),
        centroide_direito=(stats.centroide_direito[0] * sx, stats.centroide_direito[1] * sy),
        centroide_esquerdo=(stats.centroide_esquerdo[0] * sx, stats.centroide_esquerdo[1] * sy),
        labels=labels,
        clean_mask=(labels > 0).astype(np.uint8),
    )


def render_overlay(img: np.ndarray, stats: LungStats | None, alpha: float = 0.45) -> np.ndarray:
    """Overlay RGB: pulmão direito em azul, esquerdo em verde, com letras D e E."""
    rgb = cv2.cvtColor(to_gray(img), cv2.COLOR_GRAY2RGB)
    if stats is None:
        return rgb
    color = np.zeros_like(rgb)
    color[stats.labels == 1] = COLOR_DIREITO
    color[stats.labels == 2] = COLOR_ESQUERDO
    inside = stats.labels > 0
    out = rgb.copy()
    out[inside] = ((1 - alpha) * rgb[inside] + alpha * color[inside]).astype(np.uint8)

    scale = max(rgb.shape[:2]) / 400
    thick = max(1, round(scale * 2))
    for letter, (cx, cy) in (("D", stats.centroide_direito), ("E", stats.centroide_esquerdo)):
        (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = (int(cx - tw / 2), int(cy + th / 2))
        cv2.putText(
            out, letter, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA
        )
    return out


def describe(stats: LungStats | None) -> str:
    if stats is None:
        return (
            "Não foi possível identificar dois pulmões nesta imagem. "
            "Verifique se é um raio-X de tórax frontal (PA)."
        )
    total = stats.area_direito + stats.area_esquerdo
    return (
        f"Pulmão direito: {100 * stats.area_direito / total:.1f}% da área pulmonar\n"
        f"Pulmão esquerdo: {100 * stats.area_esquerdo / total:.1f}% da área pulmonar\n"
        f"Razão D/E: {stats.razao:.3f}"
    )
