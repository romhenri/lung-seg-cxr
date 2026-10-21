"""Escolha do dispositivo de execução."""

import logging

import torch

log = logging.getLogger(__name__)


def get_device() -> torch.device:
    """cuda se disponível, senão mps (Apple Silicon), senão cpu."""
    if torch.cuda.is_available():
        name = "cuda"
    elif torch.backends.mps.is_available():
        name = "mps"
    else:
        name = "cpu"
    log.info("dispositivo: %s", name)
    return torch.device(name)
