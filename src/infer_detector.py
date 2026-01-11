import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from datasets import load_detection_dataset
from detection_transforms import get_transform
from utils import get_default_device


def draw_ground_truth(image, target, class_names, color="green"):
    draw = ImageDraw.Draw(image)
    boxes = target["boxes"]
    labels = target["labels"]
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.tolist()
        name = class_names[label] if label < len(class_names) else str(label)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1), f"GT {name}", fill=color)
    return image


def draw_predictions(image, preds, class_names, score_threshold=0.5, topk=None):
    draw = ImageDraw.Draw(image)
    boxes = preds["boxes"]
    labels = preds["labels"]
    scores = preds["scores"]

    keep = scores >= score_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if topk is not None and boxes.numel():
        idx = torch.argsort(scores, descending=True)[:topk]
        boxes = boxes[idx]
        labels = labels[idx]
        scores = scores[idx]

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        name = class_names[label] if label < len(class_names) else str(label)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{name} {score:.2f}", fill="red")
    return image


def load_images(path):
    path = Path(path)
    if path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return [p for p in path.iterdir() if p.suffix.lower() in exts]
    return [path]


def main():
    parser = argparse.ArgumentParser(description="Inference on images")
    parser.add_argument("--data", default="archive", help="Dataset root (for class names)")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--images", required=True, help="Image or folder")
    parser.add_argument("--output", default="runs/preds", help="Output folder")
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=None, help="Draw only top-K predictions per image")
    parser.add_argument("--draw-gt", action="store_true", help="Overlay ground-truth boxes in green")
    parser.add_argument("--device", default=get_default_device())
    parser.add_argument("--model", choices=["fasterrcnn", "ssdlite"], default=None)
    args = parser.parse_args()

    dataset = load_detection_dataset(args.data, transforms=None)
    class_names = dataset.class_names
    name_to_idx = None
    if hasattr(dataset, "image_files"):
        name_to_idx = {Path(name).name: idx for idx, name in enumerate(dataset.image_files)}

    from train_detector import build_model

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_name = args.model or checkpoint.get("model_name", "fasterrcnn")
    model = build_model(len(class_names), model_name)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in load_images(args.images):
        image = Image.open(image_path).convert("RGB")
        tensor = get_transform(train=False)(image, {})[0].unsqueeze(0).to(args.device)
        with torch.no_grad():
            pred = model(tensor)[0]
            pred = {k: v.cpu() for k, v in pred.items()}
        out_image = draw_predictions(image.copy(), pred, class_names, args.score, args.topk)
        if args.draw_gt and name_to_idx and image_path.name in name_to_idx:
            _, target = dataset[name_to_idx[image_path.name]]
            out_image = draw_ground_truth(out_image, target, class_names)
        out_image.save(output_dir / image_path.name)

    print(f"Predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
