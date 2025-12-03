"""Build ImageFolder-style crops dataset from the detection subset."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO

from src.detr.utils import ensure_dir, load_json


def safe_name(x: str) -> str:
    return x.strip().replace("/", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--split", type=str, choices=["train", "val"], default="train")
    ap.add_argument("--min_area", type=float, default=1024.0)
    ap.add_argument("--pad", type=float, default=0.1)
    ap.add_argument("--max_per_class", type=int, default=5000)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    meta = load_json(data_root / "meta.json")
    classes = [safe_name(c) for c in meta["classes"]]

    ann_path = data_root / f"annotations/instances_{args.split}.json"
    img_dir = data_root / f"images/{args.split}2017"
    coco = COCO(str(ann_path))

    split_out = ensure_dir(out_root / args.split)
    class_dirs = {i: ensure_dir(split_out / classes[i]) for i in range(len(classes))}
    counts = {i: 0 for i in range(len(classes))}

    for img_id in coco.imgs.keys():
        img_info = coco.loadImgs([img_id])[0]
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        for a in anns:
            lab = int(a["category_id"]) - 1
            if counts[lab] >= args.max_per_class:
                continue

            x, y, w, h = a["bbox"]
            if w * h < args.min_area:
                continue

            px, py = args.pad * w, args.pad * h
            x0 = max(0, int(x - px))
            y0 = max(0, int(y - py))
            x1 = min(W, int(x + w + px))
            y1 = min(H, int(y + h + py))
            if (x1 - x0) < 4 or (y1 - y0) < 4:
                continue

            crop = img.crop((x0, y0, x1, y1))
            out_path = class_dirs[lab] / f"{int(img_id)}_{int(a['id'])}.jpg"
            crop.save(out_path, quality=95)
            counts[lab] += 1

    print("[done] crops ->", split_out)
    for i, c in enumerate(classes):
        print(f"  {c}: {counts[i]}")


if __name__ == "__main__":
    main()
