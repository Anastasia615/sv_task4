import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from datasets import CroppedBoxDataset, load_detection_dataset
from utils import get_default_device, set_seed


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class VGGSmall(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class LabelShiftDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label - 1


def split_indices(length, val_split, seed):
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    split = int(length * (1 - val_split))
    return indices[:split], indices[split:]


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="LeNet-5 / VGG-like classification on cropped boxes")
    parser.add_argument("--data", default="archive", help="Dataset root")
    parser.add_argument("--model", choices=["lenet", "vgg"], default="lenet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-image", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--device", default=get_default_device())
    args = parser.parse_args()

    set_seed(args.seed)

    tfm = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    dataset = load_detection_dataset(args.data, transforms=None)
    crops = CroppedBoxDataset(dataset, transforms=tfm, max_per_image=args.max_per_image)

    if args.max_samples and len(crops) > args.max_samples:
        indices = list(range(len(crops)))
        random.shuffle(indices)
        crops = Subset(crops, indices[: args.max_samples])

    crops = LabelShiftDataset(crops)

    train_idx, val_idx = split_indices(len(crops), args.val_split, args.seed)
    train_set = Subset(crops, train_idx)
    val_set = Subset(crops, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = len(dataset.class_names) - 1
    if args.model == "lenet":
        model = LeNet5(num_classes=num_classes)
    else:
        model = VGGSmall(num_classes=num_classes)

    model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        acc = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {loss:.4f} - val_acc: {acc:.4f}")


if __name__ == "__main__":
    main()
