"""Download MS COCO 2017 train/val images and annotations.

Usage:
  python -m src.detr.prepare_coco --data_root data/coco --download
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from src.detr.utils import ensure_dir

COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[skip] exists: {dst}")
        return
    print(f"[download] {url} -> {dst}")
    urlretrieve(url, dst)


def unzip(zip_path: Path, out_dir: Path) -> None:
    print(f"[unzip] {zip_path} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="e.g. data/coco")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root)
    ensure_dir(root / "images")
    ensure_dir(root / "annotations")

    if not args.download:
        print("Nothing to do (pass --download).")
        return

    tmp = ensure_dir(root / "_downloads")

    train_zip = tmp / "train2017.zip"
    val_zip = tmp / "val2017.zip"
    ann_zip = tmp / "annotations_trainval2017.zip"

    download(COCO_URLS["train_images"], train_zip)
    download(COCO_URLS["val_images"], val_zip)
    download(COCO_URLS["annotations"], ann_zip)

    unzip(train_zip, root / "images")
    unzip(val_zip, root / "images")
    unzip(ann_zip, root)

    print("[done] COCO in:", root)


if __name__ == "__main__":
    main()
