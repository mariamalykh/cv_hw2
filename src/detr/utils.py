from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from collections.abc import Mapping, Sequence


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def to_device(x, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)

    if isinstance(x, Mapping):
        return {k: to_device(v, device) for k, v in x.items()}

    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        return [to_device(v, device) for v in x]

    return x


def unwrap_loss_dict(outputs: Any) -> Dict[str, float]:
    d = getattr(outputs, "loss_dict", None)
    if d is None:
        return {}
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            out[k] = float(v.detach().cpu().item())
        except Exception:
            try:
                out[k] = float(v)
            except Exception:
                pass
    return out


def clip_grad_norm_(params: Iterable[torch.nn.Parameter], max_norm: float) -> float:
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm).item())


def build_param_groups(model: torch.nn.Module, lr: float, lr_backbone: float, weight_decay: float) -> List[Dict[str, Any]]:
    backbone, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone.append(p)
        else:
            rest.append(p)

    return [
        {"params": rest, "lr": lr, "weight_decay": weight_decay},
        {"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay},
    ]
