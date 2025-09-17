#!/usr/bin/env python3
"""
DBSCAN clustering on an image frame and visualization.

- Converts input image to grayscale and thresholds to select foreground points
- Runs sklearn.cluster.DBSCAN in pixel space (x,y)
- Colors each cluster and overlays on original image
- Optionally draws cluster bounding boxes and centroids

Usage (from repo root):
  python scripts/dbscan.py --image path/to/frame.png --eps 3.0 --min-samples 10 --th 32 --save out.png

Notes:
- eps is in pixels; scale according to object size
- For dense blobs, connected components may be faster than DBSCAN
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import os
from typing import List, Tuple

import numpy as np
import cv2
from sklearn.cluster import DBSCAN


@dataclass
class ClusterStat:
    label: int
    size: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # x, y, w, h


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def extract_points(mask: np.ndarray) -> np.ndarray:
    y, x = np.where(mask)
    if x.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.stack([x, y], axis=1).astype(np.float32)
    return pts


def compute_cluster_stats(labels: np.ndarray, pts: np.ndarray) -> List[ClusterStat]:
    stats: List[ClusterStat] = []
    unique_labels = [l for l in np.unique(labels) if l != -1]
    for lab in unique_labels:
        idx = labels == lab
        if not np.any(idx):
            continue
        p = pts[idx]
        size = int(p.shape[0])
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        xmin, ymin = np.floor(p.min(axis=0)).astype(int)
        xmax, ymax = np.ceil(p.max(axis=0)).astype(int)
        stats.append(ClusterStat(
            label=int(lab),
            size=size,
            centroid=(float(cx), float(cy)),
            bbox=(int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1)),
        ))
    return stats


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    # Map each label to a distinct color; -1 (noise) -> gray
    max_label = labels.max() if labels.size else -1
    rng = np.random.default_rng(1234)
    palette = np.zeros((max(max_label + 1, 1), 3), dtype=np.uint8)
    if max_label >= 0:
        palette[:] = rng.integers(0, 255, size=(max_label + 1, 3), dtype=np.uint8)
    return palette


def overlay_clusters(img: np.ndarray, pts: np.ndarray, labels: np.ndarray,
                     alpha: float = 0.6, point_size: int = 1,
                     draw_boxes: bool = False) -> Tuple[np.ndarray, List[ClusterStat]]:
    h, w = img.shape[:2]
    if img.ndim == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        base = img.copy()

    overlay = base.copy()
    palette = colorize_labels(labels)

    # Draw points
    for (x, y), lab in zip(pts.astype(int), labels):
        if lab == -1:
            color = (128, 128, 128)
        else:
            color = tuple(int(c) for c in palette[lab].tolist())
        cv2.circle(overlay, (int(x), int(y)), point_size, color, -1, lineType=cv2.LINE_AA)

    out = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

    # Stats and optional boxes
    stats = compute_cluster_stats(labels, pts)
    if draw_boxes:
        for st in stats:
            x, y, w_box, h_box = st.bbox
            cv2.rectangle(out, (x, y), (x + w_box, y + h_box), (0, 255, 0), 1)
            cv2.circle(out, (int(st.centroid[0]), int(st.centroid[1])), 2, (0, 255, 255), -1)
            cv2.putText(out, f"id={st.label} n={st.size}", (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    return out, stats


def run_dbscan_on_image(img: np.ndarray, eps: float, min_samples: int, th: int,
                        morph_open: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = to_gray(img)
    # Threshold (>= th is foreground)
    _, mask = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

    if morph_open and morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    pts = extract_points(mask > 0)
    if pts.shape[0] == 0:
        return mask, pts, np.empty((0,), dtype=int)

    # DBSCAN in (x,y) pixel space
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(pts)
    return mask, pts, labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DBSCAN clustering on an image frame and visualization.")
    p.add_argument("--image", required=True, help="Path to input image (png/jpg)")
    p.add_argument("--eps", type=float, default=3.0, help="Neighborhood radius in pixels")
    p.add_argument("--min-samples", type=int, default=10, help="Minimum points to form a cluster")
    p.add_argument("--th", type=int, default=32, help="Threshold for foreground (0-255)")
    p.add_argument("--open", type=int, default=0, help="Morphological open kernel size (0 to disable)")
    p.add_argument("--save", default=None, help="Optional output path to save visualization")
    p.add_argument("--show", action="store_true", help="Show the result in a window")
    return p.parse_args()


def main() -> None:
    
    args = parse_args()
    if not os.path.isfile(args.image):
        raise SystemExit(f"Image not found: {args.image}")

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    mask, pts, labels = run_dbscan_on_image(img, args.eps, args.min_samples, args.th, args.open)
    vis, stats = overlay_clusters(img, pts, labels)

    if args.save:
        out_path = args.save
        ok = cv2.imwrite(out_path, vis)
        if not ok:
            raise SystemExit(f"Failed to save: {out_path}")
        print(f"Saved: {out_path} | clusters: {len(stats)} | points: {pts.shape[0]}")

    if args.show or not args.save:
        win = "DBSCAN Clusters"
        cv2.imshow(win, vis)
        print(f"clusters: {len(stats)} | points: {pts.shape[0]}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
