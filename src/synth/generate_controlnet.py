"""Generate synthetic crops using Stable Diffusion + ControlNet (Canny)."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from src.detr.utils import ensure_dir


def safe_name(x: str) -> str:
    return x.strip().replace("/", "_")


def pil_to_canny(pil: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    img = np.array(pil.convert("RGB"))
    edges = cv2.Canny(img, low, high)
    edges = np.stack([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--classes", type=str, nargs="+", required=True)
    ap.add_argument("--num_per_image", type=int, default=2)
    ap.add_argument("--max_images_per_class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet", type=str, default="lllyasviel/sd-controlnet-canny")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--control_scale", type=float, default=1.0)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=384)
    args = ap.parse_args()

    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.sd_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.to(device)

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    for cls in args.classes:
        cls_in = Path(args.real_root) / safe_name(cls)
        if not cls_in.exists():
            print(f"[skip] missing: {cls_in}")
            continue
        imgs = sorted(cls_in.glob("*.jpg"))
        random.shuffle(imgs)
        imgs = imgs[: args.max_images_per_class]

        cls_out = ensure_dir(out_root / safe_name(cls))
        for img_path in tqdm(imgs, desc=f"gen {cls}"):
            base = Image.open(img_path).convert("RGB").resize((args.width, args.height))
            canny = pil_to_canny(base)

            prompt = f"a high quality photo of a {cls}, realistic, sharp, natural lighting"
            negative = "blurry, low quality, distorted, deformed, cartoon, text, watermark"

            for k in range(args.num_per_image):
                g = torch.Generator(device=device).manual_seed(random.randint(0, 10**9))
                out = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=canny,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    controlnet_conditioning_scale=args.control_scale,
                    generator=g,
                    height=args.height,
                    width=args.width,
                ).images[0]
                out.save(cls_out / f"{img_path.stem}_synth{k}.jpg", quality=95)

    print("[done] synth ->", out_root)


if __name__ == "__main__":
    main()
