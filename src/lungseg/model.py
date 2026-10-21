"""Construção da U-Net e checkpoints."""

from dataclasses import asdict
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

from .config import Config, ModelConfig, from_dict


def build_model(mcfg: ModelConfig, pretrained: bool = True) -> torch.nn.Module:
    return smp.create_model(
        mcfg.arch,
        encoder_name=mcfg.encoder,
        encoder_weights=mcfg.encoder_weights if pretrained else None,
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
