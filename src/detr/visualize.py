"""Save example visualizations of DETR predictions vs ground truth."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from src.detr.coco_dataset import CocoDetection
from src.detr.collate import build_collate_fn
from src.detr.utils import ensure_dir, load_json, to_device


def draw_boxes(img: Image.Image, boxes_xyxy: List[Tuple[float, float, float, float]], labels: List[str], color: Tuple[int, int, int]) -> Image.Image:
    im = img.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for box, lab in zip(boxes_xyxy, labels):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        if lab:
            draw.text((x0 + 2, y0 + 2), lab, fill=color, font=font)
    return im


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="facebook/detr-resnet-50")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num_images", type=int, default=30)
    ap.add_argument("--score_thr", type=float, default=0.3)
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
    viz_dir = ensure_dir(out_dir / "viz")

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
    idxs = idxs[: args.num_images]
    subset = [ds[i] for i in idxs]

    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=build_collate_fn(image_processor, train=False))

    for i, batch in enumerate(tqdm(loader, desc="viz")):
        img_id = int(batch["labels"][0]["image_id"].detach().cpu().item())
        img_info = ds.coco.loadImgs([img_id])[0]
        img_path = val_img_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")

        gt_boxes, gt_labels = [], []
        for a in ds.coco.loadAnns(ds.coco.getAnnIds(imgIds=[img_id])):
            x, y, w, h = a["bbox"]
            gt_boxes.append((x, y, x + w, y + h))
            gt_labels.append(str(classes[int(a["category_id"]) - 1]))

        target_sizes = torch.stack([lbl["orig_size"] for lbl in batch["labels"]], dim=0)
        batch = to_device(batch, device)
        outputs = model(pixel_values=batch["pixel_values"], pixel_mask=batch.get("pixel_mask"))
        results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes.to(device), threshold=args.score_thr)[0]

        pred_boxes = results["boxes"].detach().cpu().numpy().tolist()
        pred_labels = [f"{id2label[int(l)]}:{float(s):.2f}" for l, s in zip(results["labels"], results["scores"])]

        im_gt = draw_boxes(img, gt_boxes, gt_labels, color=(0, 200, 0))
        im_both = draw_boxes(im_gt, pred_boxes, pred_labels, color=(220, 0, 0))
        im_both.save(viz_dir / f"{i:03d}_img{img_id}.jpg", quality=95)

    print("[done] saved to:", viz_dir)


if __name__ == "__main__":
    main()
