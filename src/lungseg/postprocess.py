"""Separação dos pulmões direito e esquerdo e razão entre as áreas."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LungStats:
    area_direito: int
    area_esquerdo: int
    razao: float  # direito / esquerdo
    centroide_direito: tuple[float, float]  # (x, y)
    centroide_esquerdo: tuple[float, float]
    labels: np.ndarray  # uint8: 0 = fundo, 1 = direito, 2 = esquerdo
    clean_mask: np.ndarray  # uint8 binária, só as duas regiões mantidas


def lung_stats(mask: np.ndarray) -> LungStats | None:
    """Mantém as duas maiores regiões conexas da máscara e identifica cada pulmão.

    Convenção radiológica: o raio-X PA é visto como se o paciente estivesse de frente
    para o observador, então a região mais à ESQUERDA da imagem é o pulmão DIREITO do
    paciente, e a mais à direita é o pulmão ESQUERDO.

    Retorna None se a máscara tiver menos de duas regiões.
    """
    n, lab, stats, cents = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    if n < 3:  # fundo + pelo menos duas regiões
        return None
    biggest = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:2]
    right, left = sorted(biggest, key=lambda i: cents[i, 0])

    labels = np.zeros(mask.shape[:2], np.uint8)
    labels[lab == right] = 1
    labels[lab == left] = 2
    area_r = int(stats[right, cv2.CC_STAT_AREA])
    area_l = int(stats[left, cv2.CC_STAT_AREA])
    return LungStats(
        area_direito=area_r,
        area_esquerdo=area_l,
        razao=area_r / area_l,
        centroide_direito=tuple(map(float, cents[right])),
        centroide_esquerdo=tuple(map(float, cents[left])),
        labels=labels,
        clean_mask=(labels > 0).astype(np.uint8),
    )
