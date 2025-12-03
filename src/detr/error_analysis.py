"""Error analysis for object detection (classification vs localization)."""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from src.detr.coco_dataset import CocoDetection
from src.detr.collate import build_collate_fn
from src.detr.utils import ensure_dir, load_json, save_json, to_device


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def draw_debug(img: Image.Image, gt: List[Tuple[np.ndarray, str]], preds: List[Tuple[np.ndarray, str, float]], title: str) -> Image.Image:
    im = img.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((5, 5), title, fill=(255, 255, 0), font=font)

    for box, lab in gt:
        x0, y0, x1, y1 = box.tolist()
        draw.rectangle([x0, y0, x1, y1], outline=(0, 200, 0), width=3)
        draw.text((x0 + 2, y0 + 2), f"GT:{lab}", fill=(0, 200, 0), font=font)

    for box, lab, score in preds:
        x0, y0, x1, y1 = box.tolist()
        draw.rectangle([x0, y0, x1, y1], outline=(220, 0, 0), width=3)
        draw.text((x0 + 2, y0 + 14), f"P:{lab} {score:.2f}", fill=(220, 0, 0), font=font)

    return im


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="facebook/detr-resnet-50")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--max_images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)

    data_root = Path(args.data_root)
    meta = load_json(data_root / "meta.json")
    classes = meta["classes"]
    num_classes = int(meta["num_classes"])
    id2label = {i: str(classes[i]) for i in range(num_classes)}
    label2id = {v: k for k, v in id2label.items()}

    out_dir = ensure_dir(args.output_dir)
    ana_dir = ensure_dir(out_dir / "analysis")
    ex_dir = ensure_dir(ana_dir / "examples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForObjectDetection.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    val_ann = data_root / "annotations/instances_val.json"
    val_img_dir = data_root / "images/val2017"
    ds = CocoDetection(val_ann, val_img_dir)

    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(args.max_images, len(idxs))]
    subset = [ds[i] for i in idxs]
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=build_collate_fn(image_processor, train=False))

    per_class = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    buckets = {"cls_error": [], "loc_error": [], "missed": [], "false_pos": []}

    for batch in tqdm(loader, desc="error analysis"):
        img_id = int(batch["labels"][0]["image_id"].detach().cpu().item())
        img_info = ds.coco.loadImgs([img_id])[0]
        img_path = val_img_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")

        gt_anns = ds.coco.loadAnns(ds.coco.getAnnIds(imgIds=[img_id]))
        gt_boxes, gt_labels = [], []
        for a in gt_anns:
            x, y, w, h = a["bbox"]
            gt_boxes.append(np.array([x, y, x + w, y + h], dtype=np.float32))
            gt_labels.append(int(a["category_id"]) - 1)

        target_sizes = torch.stack([lbl["orig_size"] for lbl in batch["labels"]], dim=0)
        batch = to_device(batch, device)
        outputs = model(pixel_values=batch["pixel_values"], pixel_mask=batch.get("pixel_mask"))
        results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes.to(device), threshold=args.score_thr)[0]

        pred_scores = results["scores"].detach().cpu().numpy().astype(np.float32)
        pred_labels = results["labels"].detach().cpu().numpy().astype(np.int64)
        pred_boxes = results["boxes"].detach().cpu().numpy().astype(np.float32)

        matched_pred = set()

        # Greedy matching: GT -> best unmatched pred by IoU
        for gbox, glab in zip(gt_boxes, gt_labels):
            best_iou, best_pi = 0.0, None
            for pi, pbox in enumerate(pred_boxes):
                if pi in matched_pred:
                    continue
                iou = iou_xyxy(gbox, pbox)
                if iou > best_iou:
                    best_iou, best_pi = iou, pi

            totals["gt"] += 1
            per_class[glab]["gt"] += 1

            if best_pi is None:
                totals["missed"] += 1
                per_class[glab]["missed"] += 1
                if len(buckets["missed"]) < 30:
                    buckets["missed"].append((img, img_id, [(gbox, id2label[glab])], []))
                continue

            matched_pred.add(best_pi)
            plab = int(pred_labels[best_pi])
            pbox = pred_boxes[best_pi]
            pscore = float(pred_scores[best_pi])

            if best_iou >= args.iou_thr:
                if plab == glab:
                    totals["tp"] += 1
                    per_class[glab]["tp"] += 1
                else:
                    totals["cls_error"] += 1
                    per_class[glab]["cls_error"] += 1
                    if len(buckets["cls_error"]) < 30:
                        buckets["cls_error"].append((img, img_id, [(gbox, id2label[glab])], [(pbox, id2label[plab], pscore)]))
            else:
                totals["loc_error"] += 1
                per_class[glab]["loc_error"] += 1
                if len(buckets["loc_error"]) < 30:
                    buckets["loc_error"].append((img, img_id, [(gbox, id2label[glab])], [(pbox, id2label[plab], pscore)]))

        # Unmatched preds => false positives
        for pi, (pbox, plab, pscore) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            if pi in matched_pred:
                continue
            totals["fp"] += 1
            per_class[int(plab)]["fp"] += 1
            if len(buckets["false_pos"]) < 30:
                buckets["false_pos"].append((img, img_id, [], [(pbox, id2label[int(plab)], float(pscore))]))

    summary = {"settings": {"iou_thr": args.iou_thr, "score_thr": args.score_thr, "max_images": args.max_images}, "totals": dict(totals), "per_class": {}}
    rows = []
    for lab in range(num_classes):
        d = per_class[lab]
        row = [id2label[lab], int(d.get("gt", 0)), int(d.get("tp", 0)), int(d.get("missed", 0)), int(d.get("cls_error", 0)), int(d.get("loc_error", 0)), int(d.get("fp", 0))]
        rows.append(row)
        summary["per_class"][id2label[lab]] = dict(zip(["gt", "tp", "missed", "cls_error", "loc_error", "fp"], row[1:]))

    save_json(ana_dir / "error_summary.json", summary)
    with (ana_dir / "error_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "gt", "tp", "missed", "cls_error", "loc_error", "fp"])
        w.writerows(rows)

    for bucket, items in buckets.items():
        for j, (img, img_id, gt_list, pr_list) in enumerate(items):
            title = f"{bucket} img_id={img_id}"
            gt_draw = [(b, lab) for b, lab in gt_list]
            pr_draw = [(b, lab, sc) for b, lab, sc in pr_list]
            example = draw_debug(img, gt_draw, pr_draw, title)
            example.save(ex_dir / f"{bucket}_{j:02d}_img{img_id}.jpg", quality=95)

    print("[done] analysis ->", ana_dir)


if __name__ == "__main__":
    main()
