"""Full training loop for DETR-family models with TensorBoard, checkpoints, profiler, and loss plots."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from src.detr.coco_dataset import CocoDetection
from src.detr.collate import build_collate_fn
from src.detr.eval_coco import evaluate_on_val
from src.detr.utils import (
    AverageMeter,
    build_param_groups,
    clip_grad_norm_,
    ensure_dir,
    load_json,
    save_json,
    set_seed,
    to_device,
    unwrap_loss_dict,
)


def plot_losses(csv_path: Path, out_png: Path) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for col in ["loss_ce", "loss_bbox", "loss_giou", "loss_total"]:
        if col in df.columns:
            ax.plot(df["global_step"], df[col], label=col)
    ax.set_xlabel("global_step")
    ax.set_ylabel("loss")
    ax.set_title("Training losses (step)")
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="facebook/detr-resnet-50")
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--eval_every_epochs", type=int, default=1)
    ap.add_argument("--score_thr", type=float, default=0.3)

    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile_wait", type=int, default=2)
    ap.add_argument("--profile_warmup", type=int, default=2)
    ap.add_argument("--profile_active", type=int, default=5)
    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = ensure_dir(args.output_dir)
    tb_dir = ensure_dir(out_dir / "tb")
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    logs_dir = ensure_dir(out_dir / "logs")
    plots_dir = ensure_dir(out_dir / "plots")
    metrics_dir = ensure_dir(out_dir / "metrics")

    meta = load_json(Path(args.data_root) / "meta.json")
    classes = meta["classes"]
    num_classes = int(meta["num_classes"])
    id2label = {i: str(classes[i]) for i in range(num_classes)}
    label2id = {v: k for k, v in id2label.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForObjectDetection.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    data_root = Path(args.data_root)
    train_ann = data_root / "annotations/instances_train.json"
    val_ann = data_root / "annotations/instances_val.json"
    train_img_dir = data_root / "images/train2017"
    val_img_dir = data_root / "images/val2017"
    if not train_img_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {train_img_dir} (create symlink or copy from COCO)")
    if not val_img_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {val_img_dir} (create symlink or copy from COCO)")

    train_ds = CocoDetection(train_ann, train_img_dir)
    val_ds = CocoDetection(val_ann, val_img_dir)

    train_ds.image_ids = train_ds.image_ids[:5000]  
    val_ds.image_ids = val_ds.image_ids[:1000]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=build_collate_fn(image_processor, train=True),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=build_collate_fn(image_processor, train=False),
    )

    optimizer = AdamW(build_param_groups(model, args.lr, args.lr_backbone, args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
    writer = SummaryWriter(log_dir=str(tb_dir))

    csv_path = logs_dir / "train_steps.csv"
    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_f,
        fieldnames=["epoch", "step", "global_step", "loss_total", "loss_ce", "loss_bbox", "loss_giou", "class_error", "lr"],
    )
    csv_writer.writeheader()

    prof = None
    if args.profile and device.type == "cuda":
        schedule = torch.profiler.schedule(wait=args.profile_wait, warmup=args.profile_warmup, active=args.profile_active, repeat=1)
        prof = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(tb_dir / "profiler")),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        prof.__enter__()

    best_map = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        meter_total = AverageMeter()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(pbar, start=1):
            target_sizes = torch.stack([lbl["orig_size"] for lbl in batch["labels"]], dim=0)
            batch = to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    pixel_mask=batch.get("pixel_mask"),
                    labels=batch["labels"],
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if prof is not None:
                prof.step()

            loss_total = float(loss.detach().cpu().item())
            loss_dict = unwrap_loss_dict(outputs)

            meter_total.update(loss_total, n=batch["pixel_values"].shape[0])
            lr_now = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{meter_total.avg:.4f}", lr=f"{lr_now:.2e}")

            writer.add_scalar("train/loss_total", loss_total, global_step)
            for k in ["loss_ce", "loss_bbox", "loss_giou", "class_error"]:
                if k in loss_dict:
                    writer.add_scalar(f"train/{k}", loss_dict[k], global_step)
            writer.add_scalar("train/grad_norm", grad_norm, global_step)
            writer.add_scalar("train/lr", lr_now, global_step)

            csv_writer.writerow({
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "loss_total": loss_total,
                "loss_ce": loss_dict.get("loss_ce", float("nan")),
                "loss_bbox": loss_dict.get("loss_bbox", float("nan")),
                "loss_giou": loss_dict.get("loss_giou", float("nan")),
                "class_error": loss_dict.get("class_error", float("nan")),
                "lr": lr_now,
            })
            global_step += 1

        torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args), "meta": meta}, ckpt_dir / "last.pt")

        if (epoch % args.eval_every_epochs) == 0:
            model.eval()
            metrics = evaluate_on_val(model, image_processor, val_loader, val_ann, device, args.score_thr)
            writer.add_scalar("val/mAP", metrics["mAP"], epoch)
            writer.add_scalar("val/mAP50", metrics["mAP50"], epoch)
            save_json(metrics_dir / "val_metrics.json", {"epoch": epoch, **metrics})

            if metrics["mAP"] > best_map:
                best_map = metrics["mAP"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args), "meta": meta}, ckpt_dir / "best.pt")
                print(f"[best] epoch={epoch} mAP={best_map:.4f}")

        csv_f.flush()
        try:
            plot_losses(csv_path, plots_dir / "loss_curves.png")
        except Exception as e:
            print("[warn] plot failed:", repr(e))

    if prof is not None:
        prof.__exit__(None, None, None)
    csv_f.close()
    writer.close()
    print("[done] outputs in:", out_dir)


if __name__ == "__main__":
    main()
