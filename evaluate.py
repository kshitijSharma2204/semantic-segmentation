"""Score checkpoints on the validation split with the standard 19-class mIoU.

For each image (batch 1): resize the input the way the model was trained, run
the model, upsample the logits bilinearly to the image's original size, argmax.
One 19x19 confusion matrix (plus a column for void predictions) is accumulated
over the whole split, and IoU = TP / (TP + FP + FN) is averaged over the classes
that have ground-truth pixels.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

from config import ROOT, add_common_args, load_labels
from make_splits import read_split
from metrics import (IGNORE, id_to_trainid_lut, new_confusion, old_batch_miou, scores,
                     trainid_names, update_confusion)
from model import UNet

# labelid34 is v1: 34 raw label IDs, inputs squashed to 256x512.
# trainid19 is v2: 19 trainIds, inputs resized to 352x1184 (close to KITTI's aspect ratio).
MODEL_TYPES = {
    "labelid34": {"n_classes": 34, "size": (256, 512)},
    "trainid19": {"n_classes": 19, "size": (352, 1184)},
}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocess(size):
    return A.Compose([A.Resize(height=size[0], width=size[1]), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


def load_model(path, device, model_type=None):
    """Returns (model, model_type, input_size). Works for v1 plain state_dicts and v2 full checkpoints."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "model" in ckpt:
        state, cfg = ckpt["model"], ckpt["config"]
        model_type = model_type or cfg["model_type"]
        size = (cfg["height"], cfg["width"])
    else:
        state = ckpt
        if model_type is None:
            n = state["conv_last.weight"].shape[0]
            model_type = {34: "labelid34", 19: "trainid19"}[n]
        size = MODEL_TYPES[model_type]["size"]
    model = UNet(n_classes=MODEL_TYPES[model_type]["n_classes"]).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, model_type, size


@torch.no_grad()
def predict(model, img, out_hw):
    """img: normalized 1x3xHxW tensor. Returns the raw logits and the logits upsampled to out_hw."""
    logits = model(img)
    return logits, F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate_split(model, model_type, size, names, data_root, device, labels, old_metric=False, progress=True):
    lut = torch.from_numpy(id_to_trainid_lut(labels)).to(device)
    n_model = MODEL_TYPES[model_type]["n_classes"]
    tf = preprocess(size)
    img_dir = os.path.join(data_root, "data_semantics/training/image_2")
    mask_dir = os.path.join(data_root, "data_semantics/training/semantic")

    conf = new_confusion(19, device)
    per_image = {}
    loss_sum, loss_pixels = 0.0, 0
    small_preds, small_masks = [], []

    for name in tqdm(names, desc="eval", disable=not progress, leave=False):
        image = np.array(Image.open(os.path.join(img_dir, name)).convert("RGB"))
        mask_id = np.array(Image.open(os.path.join(mask_dir, name)))
        img = tf(image=image)["image"].unsqueeze(0).to(device)
        gt = lut[torch.from_numpy(mask_id).to(device).long()].long()

        small_logits, logits = predict(model, img, mask_id.shape)
        pred = logits.argmax(1)[0]
        if model_type == "labelid34":
            # map predicted label IDs to trainIds; void IDs become 255 (a miss)
            pred = lut[pred].long()
        else:
            loss_sum += F.cross_entropy(logits, gt.unsqueeze(0), ignore_index=IGNORE, reduction="sum").item()
            loss_pixels += int((gt != IGNORE).sum())

        update_confusion(conf, gt, pred)
        per_image[name] = scores(update_confusion(new_confusion(19, device), gt, pred))["miou"]

        if old_metric:
            # what the old train.py validated on: labels squashed to the input size (nearest)
            small_mask = tf(image=image, mask=mask_id)["mask"].to(device).long()
            if model_type == "trainid19":
                small_mask = lut[small_mask].long()
            small_preds.append(small_logits.argmax(1)[0])
            small_masks.append(small_mask)

    result = scores(conf)
    result["per_image"] = per_image
    result["val_loss"] = loss_sum / loss_pixels if loss_pixels else None
    result["confusion"] = conf.cpu().tolist()

    if old_metric:
        # same batching as the old val_loader: batch 12, split order, weighted by batch size
        total = 0.0
        for i in range(0, len(small_preds), 12):
            p = torch.stack(small_preds[i:i + 12])
            m = torch.stack(small_masks[i:i + 12])
            if model_type == "trainid19":
                keep = m != IGNORE
                p, m = p[keep], m[keep]
            total += old_batch_miou(p, m, n_model) * len(small_preds[i:i + 12])
        result["old_batch_miou"] = total / len(small_preds)
    return result


def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "*.py"], cwd=ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_report(result, class_names, json_path, csv_path):
    with open(json_path, "w") as f:
        json.dump(result, f, indent=1)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "train_id", "iou", "gt_pixels", "pred_pixels", "tp"])
        for i, name in enumerate(class_names):
            iou = result["iou"][i]
            w.writerow([name, i, "" if iou != iou else f"{iou:.4f}",
                        result["gt_pixels"][i], result["pred_pixels"][i], result["tp"][i]])


def epoch_of(path):
    """v1 checkpoints have the epoch in the file name, v2 ones (best.pt, last.pt) store it."""
    m = re.search(r"epoch(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt.get("epoch") if "model" in ckpt else None


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Score checkpoints with the 19-class mIoU"))
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--model-type", choices=list(MODEL_TYPES), default=None,
                        help="default: read it from the checkpoint")
    parser.add_argument("--split", default=os.path.join(ROOT, "splits/val.txt"))
    parser.add_argument("--height", type=int, default=None, help="input height, default depends on model type")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = load_labels(args.data_root)
    class_names = trainid_names(labels)
    names = read_split(args.split)
    commit = git_commit()
    os.makedirs(args.out_dir, exist_ok=True)

    epochs = {p: epoch_of(p) for p in args.checkpoints}
    ckpts = sorted(args.checkpoints, key=lambda p: (epochs[p] is None, epochs[p] or 0, p))
    summary = []
    for path in tqdm(ckpts, desc="checkpoints", disable=len(ckpts) == 1):
        model, model_type, size = load_model(path, device, args.model_type)
        if args.height and args.width:
            size = (args.height, args.width)
        result = evaluate_split(model, model_type, size, names, args.data_root, device, labels,
                                old_metric=True, progress=len(ckpts) == 1)
        result.update({
            "checkpoint": os.path.relpath(os.path.abspath(path), ROOT),
            "model_type": model_type,
            "input_size": list(size),
            "split": os.path.relpath(os.path.abspath(args.split), ROOT),
            "n_images": len(names),
            "commit": commit,
            "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
            "class_names": class_names,
        })
        stem = os.path.splitext(os.path.basename(path))[0]
        write_report(result, class_names, os.path.join(args.out_dir, f"{stem}.json"),
                     os.path.join(args.out_dir, f"{stem}_classes.csv"))
        summary.append([result["checkpoint"], epochs[path], result["miou"], result["pixel_acc"],
                        result["old_batch_miou"], result["n_classes_averaged"]])
        tqdm.write(f"{stem}: mIoU {result['miou']:.4f} over {result['n_classes_averaged']} classes, "
                   f"pixel acc {result['pixel_acc']:.4f}, old per-batch metric {result['old_batch_miou']:.4f}")
        if result["classes_without_gt"]:
            tqdm.write("  no ground truth in this split for: "
                       + ", ".join(class_names[i] for i in result["classes_without_gt"]))

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "epoch", "miou", "pixel_acc", "old_batch_miou", "n_classes_averaged"])
        for row in summary:
            w.writerow(row[:2] + [f"{v:.4f}" for v in row[2:5]] + row[5:])


if __name__ == "__main__":
    main()
