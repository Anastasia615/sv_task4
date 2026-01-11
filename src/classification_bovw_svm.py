import argparse
import random

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from datasets import CroppedBoxDataset, load_detection_dataset


def create_detector():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    return cv2.ORB_create()


def extract_descriptors(images, detector):
    descriptors = []
    for img in images:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        _, desc = detector.detectAndCompute(gray, None)
        if desc is not None:
            descriptors.append(desc)
    return descriptors


def build_bovw_histogram(descriptors, kmeans):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(kmeans.n_clusters, dtype=np.float32)
    words = kmeans.predict(descriptors)
    hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
    hist = hist.astype(np.float32)
    hist /= max(hist.sum(), 1.0)
    return hist


def main():
    parser = argparse.ArgumentParser(description="BoVW + linear SVM classification")
    parser.add_argument("--data", default="archive", help="Dataset root")
    parser.add_argument("--k", type=int, default=128, help="Number of visual words")
    parser.add_argument("--max-per-image", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    dataset = load_detection_dataset(args.data, transforms=None)
    crop_dataset = CroppedBoxDataset(dataset, transforms=None, max_per_image=args.max_per_image)

    indices = list(range(len(crop_dataset)))
    random.shuffle(indices)
    indices = indices[: min(len(indices), args.max_samples)]

    images = []
    labels = []
    for idx in indices:
        img, label = crop_dataset[idx]
        images.append(img)
        labels.append(label)

    detector = create_detector()
    descriptors_list = extract_descriptors(images, detector)
    if not descriptors_list:
        raise RuntimeError("No descriptors found. Try different images or install opencv-contrib-python.")

    all_descriptors = np.vstack(descriptors_list)
    kmeans = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, batch_size=1024)
    kmeans.fit(all_descriptors)

    features = []
    for img in images:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        _, desc = detector.detectAndCompute(gray, None)
        features.append(build_bovw_histogram(desc, kmeans))

    X = np.stack(features)
    y = np.array(labels)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    clf = LinearSVC()
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
