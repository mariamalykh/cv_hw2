from __future__ import annotations

from typing import Any, Dict, List

from transformers import AutoImageProcessor

from src.detr.coco_dataset import CocoItem


def build_collate_fn(image_processor: AutoImageProcessor, train: bool) -> Any:
    def collate(batch: List[CocoItem]) -> Dict[str, Any]:
        images = [item.image for item in batch]
        annotations = []
        for item in batch:
            anns = []
            for a in item.annotations:
                a2 = dict(a)
                a2["category_id"] = int(a2["category_id"]) - 1  # 1..K -> 0..K-1
                anns.append(a2)
            annotations.append({"image_id": item.image_id, "annotations": anns})

        enc = image_processor(images=images, annotations=annotations, return_tensors="pt")
        return enc

    return collate
