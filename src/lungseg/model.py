"""Construção da U-Net e checkpoints."""

from dataclasses import asdict
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn

from .config import Config, ModelConfig, from_dict


def _conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )


class UNet3Plus(nn.Module):
    """UNet 3+ (Huang et al., 2020): cada nó do decoder recebe todas as escalas.

    Escalas mais finas do encoder entram por max pooling, a mesma escala direto e as
    mais grossas (decoder, ou o fundo do encoder) por upsampling bilinear.
    """

    # ponytail: sem deep supervision nem classification-guided module do paper;
    # adicionar se a validação mostrar falsos positivos fora do pulmão.

    def __init__(self, encoder_name: str, encoder_weights: str | None, classes: int = 1):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=3, depth=5, weights=encoder_weights)
        enc_ch = self.encoder.out_channels[1:]  # 5 escalas: /2 ... /32
        cat, n = 64, len(enc_ch)
        up = cat * n
        self.branches = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for i in range(n - 1):  # nós do decoder nas escalas 0..3
            srcs = [enc_ch[j] if j <= i or j == n - 1 else up for j in range(n)]
            self.branches.append(nn.ModuleList(_conv_bn_relu(c, cat) for c in srcs))
            self.fuse.append(_conv_bn_relu(up, up))
        self.head = nn.Conv2d(up, classes, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)[1:]
        n = len(feats)
        dec = {n - 1: feats[n - 1]}
        for i in range(n - 2, -1, -1):
            size = feats[i].shape[-2:]
            parts = []
            for j, branch in enumerate(self.branches[i]):
                src = feats[j] if j <= i else dec[j]
                if j < i:
                    src = F.max_pool2d(src, 2 ** (i - j))
                elif j > i:
                    src = F.interpolate(src, size=size, mode="bilinear", align_corners=False)
                parts.append(branch(src))
            dec[i] = self.fuse[i](torch.cat(parts, 1))
        logits = self.head(dec[0])
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(mcfg: ModelConfig, pretrained: bool = True) -> torch.nn.Module:
    weights = mcfg.encoder_weights if pretrained else None
    if mcfg.arch.lower() == "unet3plus":
        return UNet3Plus(mcfg.encoder, weights)
    return smp.create_model(
        mcfg.arch,
        encoder_name=mcfg.encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=1,
    )


def save_checkpoint(path: Path, model, cfg: Config, epoch: int, val_dice: float) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "size": cfg.data.size,
            "epoch": epoch,
            "val_dice": val_dice,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Reconstrói o modelo só a partir do arquivo. Retorna (modelo em eval, checkpoint)."""
    if not Path(path).exists():
        raise SystemExit(f"Checkpoint {path} não encontrado. Rode `lungseg train` antes.")
    ckpt = torch.load(path, map_location=device)
    model = build_model(from_dict(ckpt["config"]).model, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
