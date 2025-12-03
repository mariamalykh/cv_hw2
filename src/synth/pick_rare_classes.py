"""Pick rare classes in the COCO-subset split by counting annotations."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.detr.utils import load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, choices=["train", "val"], default="train")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    meta = load_json(data_root / "meta.json")
    classes = meta["classes"]

    coco = load_json(data_root / f"annotations/instances_{args.split}.json")
    cnt = Counter()
    for a in coco["annotations"]:
        cnt[int(a["category_id"]) - 1] += 1

    rare = cnt.most_common()[::-1][: args.topk]
    print("Rare classes:")
    for lab, n in rare:
        print(f"  - {classes[lab]}: {n}")

    print("\nAll classes:")
    for lab in range(len(classes)):
        print(f"  {classes[lab]}: {cnt[lab]}")


if __name__ == "__main__":
    main()
