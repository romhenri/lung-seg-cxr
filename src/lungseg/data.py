"""Download, pareamento, cache, split, Dataset e transforms."""

import json
import logging
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import Config

log = logging.getLogger(__name__)

IMAGE_GLOB = "**/CXR_png/*.png"
MASK_GLOB = "**/masks/*.png"
SOURCES = {"MCU": "montgomery", "CHN": "shenzhen"}

KAGGLE_AUTH_HELP = """Não foi possível baixar o dataset do Kaggle.
Configure suas credenciais de uma destas formas:
  1. Baixe o token em https://www.kaggle.com/settings (Create New Token) e salve em
     ~/.kaggle/kaggle.json  (depois: chmod 600 ~/.kaggle/kaggle.json)
  2. Exporte as variáveis KAGGLE_USERNAME e KAGGLE_KEY
Ou baixe e extraia o dataset manualmente e passe a pasta com --raw-dir."""


def resolve_raw_dir(cfg: Config) -> Path:
    """Usa raw_dir se já tiver PNGs; senão baixa com kagglehub."""
    raw = Path(cfg.data.raw_dir)
    if raw.is_dir() and any(raw.glob(IMAGE_GLOB)):
        return raw
    import kagglehub

    log.info("baixando %s com kagglehub...", cfg.data.kaggle_dataset)
    try:
        return Path(kagglehub.dataset_download(cfg.data.kaggle_dataset))
    except Exception as e:
        raise SystemExit(f"{KAGGLE_AUTH_HELP}\n\nErro original: {e}") from e


def pair_files(root: Path) -> list[tuple[str, Path, Path]]:
    """Pareia imagens e máscaras pelo nome sem extensão (máscaras do Shenzhen têm `_mask`)."""
    images = {p.stem: p for p in root.glob(IMAGE_GLOB)}
    masks = {p.stem.removesuffix("_mask"): p for p in root.glob(MASK_GLOB)}
    keys = sorted(images.keys() & masks.keys())
    log.info("imagens: %d | máscaras: %d | pares: %d", len(images), len(masks), len(keys))
    if not keys:
        dirs = sorted({str(p.parent) for p in root.rglob("*.png")})
        raise SystemExit(f"Nenhum par imagem/máscara em {root}. Pastas com PNG: {dirs}")
    return [(k, images[k], masks[k]) for k in keys]


def parse_meta(key: str) -> tuple[str, int]:
    """'CHNCXR_0001_1' -> ('shenzhen', 1). Rótulo de TB: 0 normal, 1 TB, -1 desconhecido."""
    source = SOURCES.get(key[:3], "desconhecida")
    tail = key.rsplit("_", 1)[-1]
    return source, int(tail) if tail in ("0", "1") else -1


def build_cache(pairs: list[tuple[str, Path, Path]], size: int, path: Path) -> None:
    images, masks = [], []
    for _, img_path, mask_path in tqdm(pairs, desc="cache"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        images.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        masks.append((mask > 127).astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=np.stack(images),
        masks=np.stack(masks),
        keys=np.array([k for k, _, _ in pairs]),
        paths=np.array([str(p.resolve()) for _, p, _ in pairs]),
    )
    log.info("cache salvo em %s", path)


def load_cache(cfg: Config) -> dict[str, np.ndarray]:
    path = Path(cfg.data.cache_path)
    if not path.exists():
        raise SystemExit(f"Cache {path} não encontrado. Rode `lungseg prepare` antes.")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def get_split(cfg: Config, keys: np.ndarray) -> dict[str, list[str]]:
    """Split treino/validação estratificado pela fonte, salvo e reutilizado."""
    path = cfg.out / "split.json"
    if path.exists():
        split = json.loads(path.read_text())
        if set(split["train"] + split["val"]) <= set(keys):
            return split
        log.warning("split.json não bate com o cache; gerando de novo")
    sources = [parse_meta(k)[0] for k in keys]
    train, val = train_test_split(
        list(keys), test_size=cfg.data.val_fraction, stratify=sources, random_state=cfg.seed
    )
    split = {"train": sorted(train), "val": sorted(val)}
    path.write_text(json.dumps(split, indent=1))
    return split


def prepare(cfg: Config) -> None:
    pairs = pair_files(resolve_raw_dir(cfg))
    counts = {}
    for k, _, _ in pairs:
        src = parse_meta(k)[0]
        counts[src] = counts.get(src, 0) + 1
    log.info("pares por fonte: %s", counts)
    build_cache(pairs, cfg.data.size, Path(cfg.data.cache_path))
    split = get_split(cfg, np.array([k for k, _, _ in pairs]))
    log.info("split: %d treino, %d validação", len(split["train"]), len(split["val"]))


def train_transforms() -> A.Compose:
    # NUNCA usar flip horizontal (nem rotação de 180°): espelhar a imagem troca a
    # anatomia de lado e o pulmão "direito" passaria a ser o esquerdo.
    return A.Compose(
        [
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def val_transforms() -> A.Compose:
    return A.Compose([A.Normalize(), ToTensorV2()])


class CXRDataset(Dataset):
    """Imagens cinza (N,H,W) replicadas em 3 canais para o encoder ImageNet."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, transform: A.Compose):
        self.images, self.masks, self.transform = images, masks, transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = np.repeat(self.images[i][..., None], 3, axis=2)
        out = self.transform(image=img, mask=self.masks[i])
        return out["image"], out["mask"].float().unsqueeze(0)


def subset(cache: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    """Índices do cache para uma lista de chaves."""
    index = {k: i for i, k in enumerate(cache["keys"])}
    return np.array([index[k] for k in keys])
