"""Train a CNN/ViT classifier on crops (real only vs real+synth)."""
from __future__ import annotations

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from tqdm import tqdm

from pathlib import Path
from src.detr.utils import ensure_dir, save_json, set_seed

from torch.utils.data import Dataset
from PIL import Image

def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    raise ValueError("Unknown arch. Use resnet50 or vit_b_16")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

class MappedFolderDataset(Dataset):
    def __init__(self, root: Path, split: str, class_to_idx: dict, transform=None):
        self.root = Path(root)
        self.split = split
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform
        self.samples = []

        split_dir = self.root / split
        for cls, idx in self.class_to_idx.items():
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        x = Image.open(path).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, y

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ce = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        pred = logits.argmax(1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        loss_sum += float(loss.item()) * int(y.numel())
    return {"acc": correct / max(1, total), "loss": loss_sum / max(1, total)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root", type=str, required=True)
    ap.add_argument("--synth_root", type=str, default=None)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--arch", type=str, default="vit_b_16")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    tb_dir = ensure_dir(out_dir / "tb")

    tfm_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    real_root = Path(args.real_root)

    # фиксируем список классов и их индексы по real/train
    classes = sorted([p.name for p in (real_root / "train").iterdir() if p.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    ds_train_real = MappedFolderDataset(real_root, "train", class_to_idx, tfm_train)
    ds_val = MappedFolderDataset(real_root, "val", class_to_idx, tfm_val)

    ds_train = ds_train_real
    synth_used = False

    if args.synth_root:
        synth_root = Path(args.synth_root)
        ds_train_synth = MappedFolderDataset(synth_root, "train", class_to_idx, tfm_train)
        # синтетика может быть только для части классов — это нормально
        if len(ds_train_synth) > 0:
            ds_train = ConcatDataset([ds_train_real, ds_train_synth])
            synth_used = True
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.arch, num_classes=len(classes)).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()
    writer = SummaryWriter(str(tb_dir))

    best_acc = -1.0
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            writer.add_scalar("train/loss", float(loss.item()), step)
            step += 1

        val_metrics = evaluate(model, val_loader, device)
        writer.add_scalar("val/acc", val_metrics["acc"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        print("[val]", val_metrics)

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "classes": classes, "args": vars(args)},
                ckpt_dir / "best.pt",
            )

    save_json(out_dir / "metrics.json", {
        "best_val_acc": best_acc,
        "arch": args.arch,
        "epochs": args.epochs,
        "synth_used": synth_used,
        "num_classes": len(classes),
        "classes": classes,
    })
    writer.close()
    print("[done] metrics ->", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
