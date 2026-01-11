import argparse

import torch
from torch.utils.data import DataLoader, Subset

from datasets import load_detection_dataset
from detection_transforms import get_transform
from utils import collate_fn, get_default_device, mean_average_precision


def evaluate(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for images, targs in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            outputs = [{k: v.cpu() for k, v in out.items()} for out in outputs]
            preds.extend(outputs)
            targets.extend([{k: v.cpu() for k, v in t.items()} for t in targs])
    map50, ap_per_class = mean_average_precision(preds, targets, iou_threshold=iou_threshold)
    return map50, ap_per_class


def main():
    parser = argparse.ArgumentParser(description="Evaluate detector (mAP@IoU)")
    parser.add_argument("--data", default="archive", help="Dataset root")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default=get_default_device())
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of evaluation images")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--model", choices=["fasterrcnn", "ssdlite"], default=None)
    args = parser.parse_args()

    base_dataset = load_detection_dataset(args.data, transforms=get_transform(train=False))
    class_names = base_dataset.class_names
    dataset = base_dataset
    if args.max_samples is not None and args.max_samples < len(base_dataset):
        dataset = Subset(base_dataset, list(range(args.max_samples)))
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    from train_detector import build_model

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_name = args.model or checkpoint.get("model_name", "fasterrcnn")
    model = build_model(len(class_names), model_name)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)

    map50, ap_per_class = evaluate(model, data_loader, args.device, iou_threshold=args.iou)

    print(f"mAP@{args.iou:.2f}: {map50:.4f}")
    for cls_id, ap in ap_per_class.items():
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        print(f"  {name}: {ap:.4f}")


if __name__ == "__main__":
    main()
