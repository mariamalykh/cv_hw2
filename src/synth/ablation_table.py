"""Build a small ablation markdown table from runs/*/metrics.json."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.detr.utils import load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, nargs="+", required=True)
    ap.add_argument("--out_md", type=str, required=True)
    args = ap.parse_args()

    rows = []
    for r in args.runs:
        m = load_json(Path(r) / "metrics.json")
        rows.append((Path(r).name, m.get("arch", ""), m.get("synth_used", False), float(m.get("best_val_acc", 0.0)), int(m.get("num_classes", 0))))

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "| run | arch | synth_used | best_val_acc | num_classes |",
        "|---|---|---:|---:|---:|",
    ]
    for run, arch, synth_used, acc, nc in rows:
        lines.append(f"| {run} | {arch} | {str(synth_used)} | {acc:.4f} | {nc} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[done] wrote", out)


if __name__ == "__main__":
    main()
