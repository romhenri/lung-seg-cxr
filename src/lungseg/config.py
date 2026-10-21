"""Configuração: YAML -> dataclasses, com override de qualquer chave pela CLI."""

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    kaggle_dataset: str = "nikhilpandey360/chest-xray-masks-and-labels"
    raw_dir: str = "data/raw"
    cache_path: str = "data/cache_256.npz"
    size: int = 256
    val_fraction: float = 0.15


@dataclass
class ModelConfig:
    arch: str = "unet"
    encoder: str = "resnet34"
    encoder_weights: str = "imagenet"


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 16
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    outputs_dir: str = "outputs"
    threshold: float = 0.5

    @property
    def out(self) -> Path:
        path = Path(self.outputs_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


SECTIONS = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig}


def from_dict(d: dict) -> Config:
    cfg = Config(**{k: v for k, v in d.items() if k not in SECTIONS})
    for name, cls in SECTIONS.items():
        setattr(cfg, name, cls(**d.get(name, {})))
    return cfg


def load_config(path: str | Path) -> Config:
    return from_dict(yaml.safe_load(Path(path).read_text()) or {})


def _leaves(cfg: Config) -> list[tuple[object, str, type]]:
    """(objeto dono, nome, tipo) de cada chave folha. Os nomes são únicos entre seções."""
    out = []
    for f in fields(cfg):
        if f.name in SECTIONS:
            sub = getattr(cfg, f.name)
            out += [(sub, sf.name, sf.type) for sf in fields(sub)]
        else:
            out.append((cfg, f.name, f.type))
    return out


def add_override_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("overrides do config")
    for _, name, typ in _leaves(Config()):
        group.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ, default=None)


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    for owner, name, _ in _leaves(cfg):
        value = getattr(args, name, None)
        if value is not None:
            setattr(owner, name, value)
    return cfg
