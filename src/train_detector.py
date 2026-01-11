import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssdlite320_mobilenet_v3_large
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from datasets import load_detection_dataset
from detection_transforms import get_transform, move_targets_to_device
from evaluate_detector import evaluate
from utils import collate_fn, get_default_device, save_class_names, set_seed


def build_model(num_classes, model_name):
    if model_name == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    if model_name == "ssdlite":
        return ssdlite320_mobilenet_v3_large(weights=None, weights_backbone="DEFAULT", num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")


def split_indices(length, val_split, seed):
    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    split = int(length * (1 - val_split))
    return indices[:split], indices[split:]


def set_batchnorm_eval(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_one_epoch(model, optimizer, data_loader, device, freeze_bn=False):
    model.train()
    if freeze_bn:
        set_batchnorm_eval(model)
    total_loss = 0.0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / max(len(data_loader), 1)


def save_checkpoint(path, model, optimizer, scheduler, epoch, model_name):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_name": model_name,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0)


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN with transfer learning")
    parser.add_argument("--data", default="archive", help="Path to dataset root")
    parser.add_argument("--output", default="runs/sign", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=get_default_device())
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--model", choices=["fasterrcnn", "ssdlite"], default="fasterrcnn")
    parser.add_argument("--freeze-bn", action="store_true", help="Freeze BatchNorm layers")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit val samples")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_train = load_detection_dataset(args.data, transforms=get_transform(train=True))
    dataset_eval = load_detection_dataset(args.data, transforms=get_transform(train=False))

    train_idx, val_idx = split_indices(len(dataset_train), args.val_split, args.seed)
    if args.max_train_samples is not None:
        train_idx = train_idx[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_idx = val_idx[: args.max_val_samples]
    train_set = Subset(dataset_train, train_idx)
    val_set = Subset(dataset_eval, val_idx)

    data_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    data_loader_val = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    num_classes = len(dataset_train.class_names)
    save_class_names(output_dir / "class_names.json", dataset_train.class_names)

    model = build_model(num_classes, args.model)
    model.to(args.device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, args.device)

    log_path = output_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "map50"])

    freeze_bn = args.freeze_bn or (args.model == "ssdlite" and args.batch_size == 1)
    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, optimizer, data_loader, args.device, freeze_bn=freeze_bn)
        scheduler.step()

        map50 = None
        if (epoch + 1) % args.eval_every == 0:
            map50, _ = evaluate(model, data_loader_val, args.device, iou_threshold=0.5)

        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{loss:.4f}", f"{map50:.4f}" if map50 is not None else ""])

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, scheduler, epoch + 1, args.model)

    print(f"Training finished. Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
