import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def average_precision(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def mean_average_precision(preds, targets, iou_threshold=0.5, num_classes=None):
    if num_classes is None:
        max_pred = max((p["labels"].max().item() if p["labels"].numel() else 0) for p in preds)
        max_gt = max((t["labels"].max().item() if t["labels"].numel() else 0) for t in targets)
        num_classes = int(max(max_pred, max_gt) + 1)

    ap_per_class = {}
    for cls in range(1, num_classes):
        gt_by_image = {}
        total_gt = 0
        for i, target in enumerate(targets):
            mask = target["labels"] == cls
            boxes = target["boxes"][mask]
            gt_by_image[i] = {
                "boxes": boxes,
                "detected": [False] * len(boxes),
            }
            total_gt += len(boxes)

        predictions = []
        for i, pred in enumerate(preds):
            mask = pred["labels"] == cls
            for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
                predictions.append({"image_id": i, "box": box, "score": float(score)})

        predictions.sort(key=lambda x: x["score"], reverse=True)
        if total_gt == 0:
            ap_per_class[cls] = 0.0
            continue

        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))

        for idx, pred in enumerate(predictions):
            gt = gt_by_image[pred["image_id"]]
            gt_boxes = gt["boxes"]
            if gt_boxes.numel() == 0:
                fp[idx] = 1
                continue

            ious = box_iou(pred["box"].unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, best_idx = ious.max(0)
            if best_iou >= iou_threshold and not gt["detected"][best_idx]:
                tp[idx] = 1
                gt["detected"][best_idx] = True
            else:
                fp[idx] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / (total_gt + 1e-6)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
        ap = average_precision(recalls, precisions)
        ap_per_class[cls] = float(ap)

    mean_ap = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return mean_ap, ap_per_class


def save_class_names(path, class_names):
    Path(path).write_text(json.dumps(class_names, indent=2))


def load_class_names(path):
    return json.loads(Path(path).read_text())


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
