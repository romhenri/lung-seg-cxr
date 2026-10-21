"""Ponto de entrada: `lungseg <subcomando>`."""

import argparse
import logging
import os

# Operações sem suporte no MPS caem para a CPU em vez de dar erro. Precisa vir antes do torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

from .config import add_override_args, apply_overrides, load_config  # noqa: E402

COMMANDS = {
    "prepare": "baixa o dataset, pareia imagens e máscaras e gera cache e split",
    "train": "treina a U-Net e salva o melhor checkpoint",
    "evaluate": "Dice e erro da razão D/E na validação",
    "explore": "razão D/E nas máscaras anotadas por fonte e rótulo de TB",
    "predict": "segmenta uma imagem e salva o overlay",
    "app": "abre a demo Gradio em localhost:7860",
}


def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.yaml")
    add_override_args(common)

    parser = argparse.ArgumentParser(prog="lungseg", description="Segmentação de pulmões em CXR")
    sub = parser.add_subparsers(dest="command", required=True)
    parsers = {name: sub.add_parser(name, help=h, parents=[common]) for name, h in COMMANDS.items()}
    parsers["predict"].add_argument("image", help="caminho da imagem")
    parsers["predict"].add_argument("--out", help="PNG de saída (padrão: <imagem>_overlay.png)")
    parsers["app"].add_argument("--share", action="store_true", help="link público do Gradio")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = apply_overrides(load_config(args.config), args)

    if args.command == "prepare":
        from .data import prepare

        prepare(cfg)
    elif args.command == "train":
        from .train import train

        train(cfg)
    elif args.command == "evaluate":
        from .evaluate import evaluate

        evaluate(cfg)
    elif args.command == "explore":
        from .explore import explore

        explore(cfg)
    elif args.command == "predict":
        _predict(cfg, args.image, args.out)
    elif args.command == "app":
        from .app import run_app

        run_app(cfg, share=args.share)


def _predict(cfg, image: str, out: str | None) -> None:
    from pathlib import Path

    import cv2

    from .device import get_device
    from .model import load_checkpoint
    from .predict import describe, predict_image, render_overlay

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Não foi possível ler a imagem {image}")
    device = get_device()
    model, ckpt = load_checkpoint(cfg.out / "best.pt", device)
    stats = predict_image(model, img, ckpt["size"], cfg.threshold, device)
    out = Path(out or Path(image).with_name(Path(image).stem + "_overlay.png"))
    cv2.imwrite(str(out), cv2.cvtColor(render_overlay(img, stats), cv2.COLOR_RGB2BGR))
    print(describe(stats))
    print(f"overlay salvo em {out}")
