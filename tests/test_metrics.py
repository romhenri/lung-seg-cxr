import math

import torch

from lungseg.metrics import combined_loss, dice


def test_dice():
    a = torch.zeros(2, 1, 8, 8)
    a[:, :, :4] = 1
    b = 1 - a
    assert math.isclose(dice(a, a), 1.0)
    assert dice(a, b) < 1e-6
    z = torch.zeros(2, 1, 8, 8)
    assert not math.isnan(dice(z, z))


def test_loss_is_finite():
    t = torch.randint(0, 2, (2, 1, 8, 8)).float()
    assert torch.isfinite(combined_loss()(torch.randn(2, 1, 8, 8), t))
