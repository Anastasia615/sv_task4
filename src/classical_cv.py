import argparse
from pathlib import Path

import cv2
import numpy as np


def canny_edges(gray, low=50, high=150):
    return cv2.Canny(gray, low, high)


def hough_lines(edges, threshold=150):
    return cv2.HoughLines(edges, 1, np.pi / 180, threshold)


def hough_lines_probabilistic(edges, threshold=80, min_line_length=50, max_line_gap=10):
    return cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)


def hough_circles(gray):
    return cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=30,
        minRadius=5,
        maxRadius=0,
    )


def moravec_corners(gray, window_size=5, threshold=10000, max_points=200):
    gray = gray.astype(np.float32)
    pad = window_size // 2
    h, w = gray.shape
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    responses = []

    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            patch = gray[y - pad : y + pad + 1, x - pad : x + pad + 1]
            min_ssd = np.inf
            for dy, dx in shifts:
                patch_shift = gray[
                    y + dy - pad : y + dy + pad + 1,
                    x + dx - pad : x + dx + pad + 1,
                ]
                ssd = np.sum((patch - patch_shift) ** 2)
                if ssd < min_ssd:
                    min_ssd = ssd
            if min_ssd > threshold:
                responses.append((x, y, min_ssd))

    responses.sort(key=lambda v: v[2], reverse=True)
    corners = []
    min_dist = 5
    for x, y, _ in responses:
        if len(corners) >= max_points:
            break
        if all((x - cx) ** 2 + (y - cy) ** 2 > min_dist**2 for cx, cy in corners):
            corners.append((x, y))
    return corners


def harris_corners(gray, threshold=0.01):
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    corners = np.argwhere(dst > threshold * dst.max())
    return [(int(x), int(y)) for y, x in corners]


def shi_tomasi_corners(gray, max_corners=200):
    corners = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 10)
    if corners is None:
        return []
    return [(int(x), int(y)) for x, y in corners.reshape(-1, 2)]


def sift_keypoints(gray, max_points=300):
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_points)
    else:
        detector = cv2.ORB_create(nfeatures=max_points)
    keypoints = detector.detect(gray, None)
    return keypoints


def draw_points(image, points, color=(0, 255, 0)):
    for x, y in points:
        cv2.circle(image, (x, y), 3, color, -1)
    return image


def main():
    parser = argparse.ArgumentParser(description="Classic CV demos")
    parser.add_argument("--image", required=True, help="Input image")
    parser.add_argument("--outdir", default="runs/classical", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = canny_edges(gray)
    cv2.imwrite(str(outdir / "edges_canny.png"), edges)

    lines = hough_lines(edges)
    line_img = image.copy()
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(outdir / "hough_lines.png"), line_img)

    lines_fast = hough_lines_probabilistic(edges)
    line_fast_img = image.copy()
    if lines_fast is not None:
        for x1, y1, x2, y2 in lines_fast[:, 0]:
            cv2.line(line_fast_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite(str(outdir / "hough_lines_fast.png"), line_fast_img)

    circles = hough_circles(gray)
    circle_img = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            cv2.circle(circle_img, (x, y), r, (0, 255, 0), 2)
    cv2.imwrite(str(outdir / "hough_circles.png"), circle_img)

    moravec_pts = moravec_corners(gray)
    moravec_img = draw_points(image.copy(), moravec_pts, (0, 255, 255))
    cv2.imwrite(str(outdir / "moravec.png"), moravec_img)

    harris_pts = harris_corners(gray)
    harris_img = draw_points(image.copy(), harris_pts, (255, 0, 255))
    cv2.imwrite(str(outdir / "harris.png"), harris_img)

    shi_pts = shi_tomasi_corners(gray)
    shi_img = draw_points(image.copy(), shi_pts, (0, 128, 255))
    cv2.imwrite(str(outdir / "shi_tomasi.png"), shi_img)

    keypoints = sift_keypoints(gray)
    sift_img = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(str(outdir / "sift.png"), sift_img)

    print(f"Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
