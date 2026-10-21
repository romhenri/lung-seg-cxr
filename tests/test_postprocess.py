import numpy as np

from lungseg.postprocess import lung_stats


def two_lungs():
    m = np.zeros((100, 100), np.uint8)
    m[10:90, 10:40] = 1  # esquerda da imagem: 80x30 = 2400
    m[10:70, 60:90] = 1  # direita da imagem: 60x30 = 1800
    return m


def test_left_of_image_is_right_lung():
    s = lung_stats(two_lungs())
    assert s.area_direito == 2400 and s.area_esquerdo == 1800
    assert abs(s.razao - 2400 / 1800) < 1e-9
    assert s.centroide_direito[0] < s.centroide_esquerdo[0]
    assert (s.labels[50, 20], s.labels[50, 70]) == (1, 2)


def test_small_noise_is_dropped():
    m = two_lungs()
    m[95:97, 95:97] = 1
    s = lung_stats(m)
    assert s.area_direito == 2400 and s.area_esquerdo == 1800
    assert s.clean_mask[96, 96] == 0


def test_single_region_returns_none():
    m = np.zeros((50, 50), np.uint8)
    m[10:40, 10:40] = 1
    assert lung_stats(m) is None
