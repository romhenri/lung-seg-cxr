"""Análise exploratória: razão D/E nas máscaras anotadas, por fonte e rótulo de TB."""

import logging

import matplotlib.pyplot as plt
import pandas as pd

from .config import Config
from .data import load_cache, parse_meta
from .postprocess import lung_stats

log = logging.getLogger(__name__)

WARNING = (
    "AVISO: análise exploratória. Fonte e rótulo de TB se confundem com hospital, aparelho "
    "e população; as diferenças aqui não servem para conclusões clínicas."
)


def explore(cfg: Config) -> None:
    cache = load_cache(cfg)
    rows = []
    for key, mask in zip(cache["keys"], cache["masks"], strict=True):
        stats = lung_stats(mask)
        source, tb = parse_meta(str(key))
        rows.append(
            {"key": key, "fonte": source, "tb": tb, "razao": stats.razao if stats else float("nan")}
        )
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out / "explore.csv", index=False)

    summary = df.groupby(["fonte", "tb"]).razao.agg(["count", "mean", "median", "std"])
    print(summary.round(3).to_string())
    print(f"\nmáscaras sem dois pulmões separados: {df.razao.isna().sum()}")
    print(f"\n{WARNING}")

    fig, ax = plt.subplots(figsize=(7, 5))
    df.dropna().boxplot(column="razao", by=["fonte", "tb"], ax=ax)
    ax.axhline(1.0, color="r", linestyle="--", linewidth=1)
    ax.set(
        title="Razão D/E nas máscaras anotadas (tb: 0 normal, 1 TB)",
        xlabel="(fonte, tb)",
        ylabel="razão D/E",
    )
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(cfg.out / "ratio_boxplot.png", dpi=120)
    log.info("salvos explore.csv e ratio_boxplot.png em %s", cfg.out)
