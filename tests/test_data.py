import cv2
import numpy as np

from lungseg.data import CXRDataset, pair_files, parse_meta, train_transforms, val_transforms


def test_pairing(tmp_path):
    img = np.zeros((4, 4), np.uint8)
    files = [
        "a/CXR_png/CHNCXR_0001_0.png",
        "a/masks/CHNCXR_0001_0_mask.png",
        "a/CXR_png/MCUCXR_0001_0.png",
        "a/masks/MCUCXR_0001_0.png",
        "a/CXR_png/CHNCXR_0002_1.png",  # sem máscara
    ]
    for f in files:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_path / f), img)
    pairs = pair_files(tmp_path)
    assert [k for k, _, _ in pairs] == ["CHNCXR_0001_0", "MCUCXR_0001_0"]
    assert pairs[0][2].name == "CHNCXR_0001_0_mask.png"


def test_parse_meta():
    assert parse_meta("CHNCXR_0001_1") == ("shenzhen", 1)
    assert parse_meta("MCUCXR_0042_0") == ("montgomery", 0)
    assert parse_meta("XYZ_abc")[1] == -1


def test_dataset_shapes():
    images = np.random.randint(0, 255, (3, 64, 64), np.uint8)
    masks = np.random.randint(0, 2, (3, 64, 64), np.uint8)
    for tf in (train_transforms(), val_transforms()):
        x, y = CXRDataset(images, masks, tf)[0]
        assert x.shape == (3, 64, 64) and y.shape == (1, 64, 64)
        assert x.is_floating_point() and y.is_floating_point()


def test_no_horizontal_flip():
    names = [type(t).__name__ for t in train_transforms().transforms]
    assert not any("Flip" in n or "Transpose" in n or "RandomRotate90" in n for n in names)
