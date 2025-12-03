from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = box.tolist()
    return np.array([x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)], dtype=np.float32)


@dataclass
class CocoMetrics:
    map: float
    map50: float

    def as_dict(self) -> Dict[str, float]:
        return {"mAP": float(self.map), "mAP50": float(self.map50)}


@torch.no_grad()
def run_coco_eval(coco_gt: COCO, predictions: List[Dict[str, Any]], iou_type: str = "bbox") -> CocoMetrics:
    coco_dt = coco_gt.loadRes(predictions) if len(predictions) else coco_gt.loadRes([])
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return CocoMetrics(map=float(evaluator.stats[0]), map50=float(evaluator.stats[1]))
