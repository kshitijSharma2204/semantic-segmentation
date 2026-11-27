import numpy as np
import torch

IGNORE = 255


def id_to_trainid_lut(labels):
    # 256-entry table so a whole label-ID image maps with lut[mask]
    lut = np.full(256, IGNORE, dtype=np.uint8)
    for label in labels:
        if 0 <= label.id < 256 and 0 <= label.trainId < IGNORE:
            lut[label.id] = label.trainId
    return lut


def trainid_names(labels):
    names = {label.trainId: label.name for label in labels if 0 <= label.trainId < IGNORE}
    return [names[i] for i in range(len(names))]


def new_confusion(n_classes, device="cpu"):
    # rows are ground truth, columns are predictions. The extra last column counts
    # pixels where the model predicted a void class (only the 34-class v1 model can).
    return torch.zeros(n_classes, n_classes + 1, dtype=torch.int64, device=device)


def update_confusion(conf, gt, pred):
    """gt and pred hold trainIds (255 = ignore/void) and have the same shape."""
    n = conf.shape[0]
    valid = gt != IGNORE
    gt = gt[valid].long()
    pred = pred[valid].long()
    pred = torch.where(pred >= n, n, pred)
    conf += torch.bincount(gt * (n + 1) + pred, minlength=n * (n + 1)).view(n, n + 1)
    return conf


def scores(conf):
    conf = conf.cpu().double()
    n = conf.shape[0]
    tp = conf.diagonal()
    # a void prediction stays in the row sum, so it counts as a miss for the true class
    gt_pixels = conf.sum(1)
    pred_pixels = conf[:, :n].sum(0)
    union = gt_pixels + pred_pixels - tp
    present = gt_pixels > 0
    iou = torch.where(present, tp / union.clamp(min=1), torch.tensor(float("nan"), dtype=torch.float64))
    return {
        "miou": iou[present].mean().item() if present.any() else float("nan"),
        "pixel_acc": (tp.sum() / gt_pixels.sum().clamp(min=1)).item(),
        "iou": iou.tolist(),
        "n_classes_averaged": int(present.sum()),
        "classes_without_gt": [i for i in range(n) if not present[i]],
        "tp": tp.long().tolist(),
        "gt_pixels": gt_pixels.long().tolist(),
        "pred_pixels": pred_pixels.long().tolist(),
    }


def old_batch_miou(preds, labels, n_classes):
    # The metric the first train.py logged: IoU per class within one batch, nanmean
    # over the classes that show up in that batch. Kept only to explain the old plots.
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    for cls in range(n_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / max(union, 1))
    return np.nanmean(ious)
