from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from pycocotools.coco import COCO


@dataclass
class CocoItem:
    image_id: int
    image_path: Path
    image: Image.Image
    annotations: List[Dict[str, Any]]


class CocoDetection:
    def __init__(self, ann_path: str | Path, images_dir: str | Path):
        self.ann_path = Path(ann_path)
        self.images_dir = Path(images_dir)
        self.coco = COCO(str(self.ann_path))
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> CocoItem:
        image_id = int(self.image_ids[idx])
        img_info = self.coco.loadImgs([image_id])[0]
        img_path = self.images_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)

        clean = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            clean.append(a)

        return CocoItem(image_id=image_id, image_path=img_path, image=image, annotations=clean)
