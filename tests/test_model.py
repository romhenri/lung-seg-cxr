import torch

from lungseg.config import ModelConfig
from lungseg.model import build_model


def test_unet3plus_output_shape():
    model = build_model(ModelConfig(arch="unet3plus"), pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)
