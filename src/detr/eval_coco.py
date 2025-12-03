"""COCO evaluation (mAP/mAP50) for a trained DETR model checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from pycocotools.coco import COCO

from src.detr.coco_dataset import CocoDetection
from src.detr.collate import build_collate_fn
from src.detr.eval_utils import run_coco_eval, xyxy_to_xywh
from src.detr.utils import ensure_dir, load_json, save_json, to_device


@torch.no_grad()
def evaluate_on_val(
    model: torch.nn.Module,
    image_processor: AutoImageProcessor,
    val_loader: DataLoader,
    val_ann_path: Path,
    device: torch.device,
    score_thr: float,
) -> Dict[str, float]:
    coco_gt = COCO(str(val_ann_path))
    predictions: List[Dict[str, Any]] = []

    for batch in tqdm(val_loader, desc="eval"):
        target_sizes = torch.stack([lbl["orig_size"] for lbl in batch["labels"]], dim=0)
        image_ids = [int(lbl["image_id"].detach().cpu().item()) for lbl in batch["labels"]]

        batch = to_device(batch, device)
        outputs = model(pixel_values=batch["pixel_values"], pixel_mask=batch.get("pixel_mask"))
        results = image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes.to(device), threshold=score_thr
        )

        for img_id, r in zip(image_ids, results):
            scores = r["scores"].detach().cpu().numpy()
            labels = r["labels"].detach().cpu().numpy()  # 0..K-1
            boxes = r["boxes"].detach().cpu().numpy()    # xyxy absolute

            for score, label, box in zip(scores, labels, boxes):
                predictions.append(
                    {
                        "image_id": int(img_id),
                        "category_id": int(label) + 1,  # 1..K as in subset gt json
                        "bbox": xyxy_to_xywh(np.array(box)).tolist(),
                        "score": float(score),
                    }
                )

    metrics = run_coco_eval(coco_gt, predictions)
    return metrics.as_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="facebook/detr-resnet-50")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--score_thr", type=float, default=0.3)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = ensure_dir(args.output_dir)
    metrics_dir = ensure_dir(out_dir / "metrics")

    meta = load_json(data_root / "meta.json")
    classes = meta["classes"]
    num_classes = int(meta["num_classes"])
    id2label = {i: str(classes[i]) for i in range(num_classes)}
    label2id = {v: k for k, v in id2label.items()}

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
    val_ds = CocoDetection(val_ann, val_img_dir)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=build_collate_fn(image_processor, train=False),
    )

    metrics = evaluate_on_val(model, image_processor, val_loader, val_ann, device, args.score_thr)
    save_json(metrics_dir / "val_metrics.json", metrics)
    print("[metrics]", metrics)
    print("[done] written to:", metrics_dir / "val_metrics.json")


if __name__ == "__main__":
    main()
