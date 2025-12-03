"""Create a COCO subset with a chosen list of classes.

- Filters annotations/images to contain at least one instance of the selected classes.
- Remaps category IDs to 1..K for the subset JSON.
- Writes `meta.json` with mapping and class names.

Usage:
  python -m src.detr.prepare_coco_subset \
    --coco_root data/coco \
    --out_root data/coco_subset \
    --classes person bicycle car motorcycle bus train truck boat "traffic light" "fire hydrant"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from src.detr.utils import load_json, save_json, ensure_dir


def normalize_name(x: str) -> str:
    return " ".join(x.strip().lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--classes", type=str, nargs="+", required=True, help="COCO category names, >=10")
    ap.add_argument("--train_ann", type=str, default="annotations/instances_train2017.json")
    ap.add_argument("--val_ann", type=str, default="annotations/instances_val2017.json")
    args = ap.parse_args()

    coco_root = Path(args.coco_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root / "annotations")

    classes = [normalize_name(c) for c in args.classes]
    if len(classes) < 10:
        raise ValueError("Need at least 10 classes for the assignment.")

    train = load_json(coco_root / args.train_ann)
    val = load_json(coco_root / args.val_ann)

    coco_name_to_id = {normalize_name(c["name"]): int(c["id"]) for c in train["categories"]}
    missing = [c for c in classes if c not in coco_name_to_id]
    if missing:
        raise ValueError(f"Classes not found in COCO categories: {missing}")

    selected_coco_cat_ids = [coco_name_to_id[c] for c in classes]
    selected_set = set(selected_coco_cat_ids)

    # Remap to 1..K
    new_id_by_coco_id = {coco_id: i + 1 for i, coco_id in enumerate(selected_coco_cat_ids)}
    new_categories = []
    for coco_id in selected_coco_cat_ids:
        name = next(c["name"] for c in train["categories"] if int(c["id"]) == coco_id)
        new_categories.append({"id": new_id_by_coco_id[coco_id], "name": name, "supercategory": ""})

    def filter_split(coco: Dict) -> Dict:
        anns = [a for a in coco["annotations"] if int(a["category_id"]) in selected_set]
        for a in anns:
            a["category_id"] = new_id_by_coco_id[int(a["category_id"])]

        image_ids = {int(a["image_id"]) for a in anns}
        images = [im for im in coco["images"] if int(im["id"]) in image_ids]
        image_ids = {int(im["id"]) for im in images}
        anns = [a for a in anns if int(a["image_id"]) in image_ids]

        return {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": images,
            "annotations": anns,
            "categories": new_categories,
        }

    train_subset = filter_split(train)
    val_subset = filter_split(val)

    save_json(out_root / "annotations/instances_train.json", train_subset)
    save_json(out_root / "annotations/instances_val.json", val_subset)

    meta = {
        "classes": [c for c in args.classes],
        "num_classes": len(classes),
        "coco_category_ids_original": selected_coco_cat_ids,
        "subset_category_id_from_coco": new_id_by_coco_id,
        "note": "subset JSON uses category_id 1..K; training converts to 0..K-1; evaluation converts back to 1..K.",
    }
    save_json(out_root / "meta.json", meta)

    print("[done] subset written to:", out_root)
    print("  train images:", len(train_subset["images"]), "annotations:", len(train_subset["annotations"]))
    print("  val images  :", len(val_subset["images"]), "annotations:", len(val_subset["annotations"]))
    print("  num classes :", len(classes))


if __name__ == "__main__":
    main()
