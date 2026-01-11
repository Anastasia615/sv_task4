import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def _find_image_path(root, filename):
    root = Path(root)
    candidates = list(root.rglob(filename))
    if candidates:
        return candidates[0]
    stem = Path(filename).stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        matches = list(root.rglob(stem + ext))
        if matches:
            return matches[0]
    return None


def _find_csv_annotations(root):
    root = Path(root)
    candidates = list(root.rglob("annotations.csv"))
    return candidates[0] if candidates else None


class CSVDetectionDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        csv_path = _find_csv_annotations(root)
        if csv_path is None:
            raise ValueError("annotations.csv not found")

        self.image_to_ann = {}
        class_names = set()
        image_order = []

        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename") or row.get("file") or row.get("image")
                if not filename:
                    continue
                if filename not in self.image_to_ann:
                    self.image_to_ann[filename] = []
                    image_order.append(filename)

                label = row.get("class") or row.get("label")
                if not label:
                    continue

                try:
                    xmin = float(row["xmin"])
                    ymin = float(row["ymin"])
                    xmax = float(row["xmax"])
                    ymax = float(row["ymax"])
                except (KeyError, ValueError):
                    continue

                class_names.add(label)
                self.image_to_ann[filename].append((label, [xmin, ymin, xmax, ymax]))

        if not image_order:
            raise ValueError("No valid annotations in CSV")

        class_list = sorted(class_names)
        self.class_names = ["background"] + class_list
        self.class_map = {name: idx + 1 for idx, name in enumerate(class_list)}
        self.image_files = image_order

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = _find_image_path(self.root, filename)
        if image_path is None:
            raise FileNotFoundError(f"Image not found: {filename}")

        image = Image.open(image_path).convert("RGB")

        ann_list = self.image_to_ann.get(filename, [])
        boxes = []
        labels = []
        for label, box in ann_list:
            boxes.append(box)
            labels.append(self.class_map[label])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target


class CroppedBoxDataset(Dataset):
    def __init__(self, detection_dataset, transforms=None, max_per_image=None, min_box_size=5):
        self.dataset = detection_dataset
        self.transforms = transforms
        self.items = []
        for idx in range(len(detection_dataset)):
            _, target = detection_dataset[idx]
            boxes = target["boxes"]
            labels = target["labels"]
            count = 0
            for box, label in zip(boxes, labels):
                if max_per_image is not None and count >= max_per_image:
                    break
                x1, y1, x2, y2 = box.tolist()
                if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                    continue
                self.items.append((idx, [x1, y1, x2, y2], int(label)))
                count += 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_idx, box, label = self.items[idx]
        image, _ = self.dataset[image_idx]
        if isinstance(image, torch.Tensor):
            image = image.mul(255).byte().permute(1, 2, 0).numpy()
            image = Image.fromarray(image)
        crop = image.crop(box)
        if self.transforms:
            crop = self.transforms(crop)
        return crop, label


def load_detection_dataset(root, transforms=None):
    return CSVDetectionDataset(root, transforms=transforms)
