"""Make the README images and GIF for a checkpoint.

- the test frame 000079_10, side by side and as an overlay (same frame as the v1 images)
- the best, median and worst validation images by per-image mIoU, next to the ground truth
- an overlay GIF of the raw drive, shrunk until it is under --gif-mb
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import ROOT, add_common_args, load_labels
from evaluate import load_model, predict, preprocess
from metrics import id_to_trainid_lut
from visualize import class_colors


def palette(colors):
    pal = np.zeros((256, 3), dtype=np.uint8)
    for k, v in colors.items():
        pal[k] = v
    return pal


def segment(model, tf, image, device):
    img = tf(image=image)["image"].unsqueeze(0).to(device)
    _, logits = predict(model, img, image.shape[:2])
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def overlay(image, color, alpha=0.6):
    return ((1 - alpha) * image + alpha * color).astype(np.uint8)


def panels(images, titles, path):
    plt.figure(figsize=(12, 2.2 * len(images)))
    for i, (im, title) in enumerate(zip(images, titles)):
        plt.subplot(len(images), 1, i + 1)
        plt.imshow(im)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()


def save_gif(frames, path, width, fps):
    h, w = frames[0].shape[:2]
    size = (width, round(h * width / w))
    pil = [Image.fromarray(f).resize(size, Image.BILINEAR).quantize(colors=128, method=Image.Quantize.MEDIANCUT)
           for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return os.path.getsize(path) / 2**20


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Make README media for a checkpoint"))
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "outputs/v2/checkpoints/best.pt"))
    parser.add_argument("--eval-json", default=os.path.join(ROOT, "outputs/v2/eval/best.json"),
                        help="evaluate.py output for this checkpoint, used to pick the val images")
    parser.add_argument("--frame-dir",
                        default=os.path.join(ROOT, "data/raw/2011_09_26/2011_09_26_drive_0018_sync/image_02/data"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "results/v2"))
    parser.add_argument("--gif-mb", type=float, default=8.0)
    parser.add_argument("--gif-step", type=int, default=2, help="use every n-th raw frame")
    parser.add_argument("--gif-fps", type=int, default=5)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = load_labels(args.data_root)
    model, model_type, size = load_model(args.checkpoint, device)
    tf = preprocess(size)
    pal = palette(class_colors(labels, model_type))
    lut = id_to_trainid_lut(labels)
    os.makedirs(args.out_dir, exist_ok=True)
    sem = os.path.join(args.data_root, "data_semantics")

    # 1. the same test frame as the v1 images in results/
    image = np.array(Image.open(os.path.join(sem, "testing/image_2/000079_10.png")).convert("RGB"))
    color = pal[segment(model, tf, image, device)]
    panels([image, color], ["input (test image 000079_10)", "prediction"],
           os.path.join(args.out_dir, "000079_10_result.png"))
    Image.fromarray(overlay(image, color)).save(os.path.join(args.out_dir, "000079_10_overlay.png"))

    # 2. best, median and worst validation images
    with open(args.eval_json) as f:
        per_image = json.load(f)["per_image"]
    ranked = sorted(per_image, key=per_image.get)
    picks = {"worst": ranked[0], "median": ranked[len(ranked) // 2], "best": ranked[-1]}
    for tag, name in picks.items():
        image = np.array(Image.open(os.path.join(sem, "training/image_2", name)).convert("RGB"))
        gt = lut[np.array(Image.open(os.path.join(sem, "training/semantic", name)))]
        pred = segment(model, tf, image, device)
        panels([image, pal[gt], pal[pred]],
               [f"{name}, {tag} val image (mIoU {per_image[name]:.3f})", "ground truth (black = ignored)", "prediction"],
               os.path.join(args.out_dir, f"val_{tag}.png"))
        print(f"{tag}: {name} mIoU {per_image[name]:.4f}")

    # 3. overlay GIF of the raw drive
    paths = sorted(glob.glob(os.path.join(args.frame_dir, "*.png")))[::args.gif_step]
    frames = []
    for p in tqdm(paths, desc="drive frames"):
        image = np.array(Image.open(p).convert("RGB"))
        frames.append(overlay(image, pal[segment(model, tf, image, device)]))
    gif_path = os.path.join(args.out_dir, "drive_0018_overlay.gif")
    width = 620
    mb = save_gif(frames, gif_path, width, args.gif_fps)
    while mb > args.gif_mb and width > 200:
        width = int(width * 0.85)
        mb = save_gif(frames, gif_path, width, args.gif_fps)
    print(f"GIF: {len(frames)} frames at {width} px wide, {mb:.1f} MB -> {gif_path}")


if __name__ == "__main__":
    main()
