import argparse
from pathlib import Path

import cv2
import numpy as np


def lucas_kanade_flow(img1, img2, max_corners=200):
    feature_params = dict(maxCorners=max_corners, qualityLevel=0.01, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)
    if p0 is None:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)
    return p0, p1, st


def horn_schunck_flow(img1, img2, alpha=1.0, iterations=100):
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    kernel_x = np.array([[-1, 1], [-1, 1]], dtype=np.float32) * 0.25
    kernel_y = np.array([[-1, -1], [1, 1]], dtype=np.float32) * 0.25
    kernel_t = np.ones((2, 2), dtype=np.float32) * 0.25

    Ix = cv2.filter2D(img1, -1, kernel_x) + cv2.filter2D(img2, -1, kernel_x)
    Iy = cv2.filter2D(img1, -1, kernel_y) + cv2.filter2D(img2, -1, kernel_y)
    It = cv2.filter2D(img2, -1, kernel_t) - cv2.filter2D(img1, -1, kernel_t)

    u = np.zeros_like(img1)
    v = np.zeros_like(img1)

    avg_kernel = np.array(
        [
            [1 / 12, 1 / 6, 1 / 12],
            [1 / 6, 0, 1 / 6],
            [1 / 12, 1 / 6, 1 / 12],
        ],
        dtype=np.float32,
    )

    for _ in range(iterations):
        u_avg = cv2.filter2D(u, -1, avg_kernel)
        v_avg = cv2.filter2D(v, -1, avg_kernel)
        deriv = (Ix * u_avg + Iy * v_avg + It) / (alpha ** 2 + Ix ** 2 + Iy ** 2 + 1e-6)
        u = u_avg - Ix * deriv
        v = v_avg - Iy * deriv

    return u, v


def flow_to_hsv(u, v):
    magnitude = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude * 15, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main():
    parser = argparse.ArgumentParser(description="Optical flow: Lucas-Kanade and Horn-Schunck")
    parser.add_argument("--image1", required=True, help="First frame")
    parser.add_argument("--image2", required=True, help="Second frame")
    parser.add_argument("--out", default="runs/flow.png", help="Output image (HS flow)")
    parser.add_argument("--out-lk", default="runs/flow_lk.png", help="Output image (LK flow)")
    args = parser.parse_args()

    img1_color = cv2.imread(args.image1)
    img2_color = cv2.imread(args.image2)
    if img1_color is None or img2_color is None:
        raise FileNotFoundError("Input images not found")

    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    lk = lucas_kanade_flow(img1, img2)
    if lk is not None:
        p0, p1, st = lk
        mask = np.zeros_like(img1_color)
        for i, (new, old) in enumerate(zip(p1[st == 1], p0[st == 1])):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            img1_color = cv2.circle(img1_color, (a, b), 3, (0, 0, 255), -1)
        lk_img = cv2.add(img1_color, mask)
        Path(args.out_lk).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out_lk, lk_img)

    u, v = horn_schunck_flow(img1, img2)
    flow_img = flow_to_hsv(u, v)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, flow_img)

    print(f"Saved HS flow to: {args.out}")
    if lk is not None:
        print(f"Saved LK flow to: {args.out_lk}")


if __name__ == "__main__":
    main()
