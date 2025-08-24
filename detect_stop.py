#!/usr/bin/env python3
"""
STOP sign detection using OpenCV (color + simple shape filter).
- Detect red color in HSV (two bands).
- Morphological filtering to reduce noise.
- Choose the best contour candidate (near-octagon, square-like, solid).
- Annotate bounding box and center, save outputs, write CSV.

Usage:
    python detect_stop.py --data ./stop_sign_dataset --out ./results --save-mask
"""
import argparse
import csv
import glob
import os
from typing import Optional, Tuple, Dict

import cv2
import numpy as np


def resize_max_dim(img: np.ndarray, max_dim: int = 1200) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def red_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """Detect red color in two HSV bands."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 35, 35], dtype=np.uint8)
    upper1 = np.array([15, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 35, 35], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # noise reduction
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)
    return mask


def select_best_candidate(contours, min_area: int, img_wh: Tuple[int, int],
                          red_mask: np.ndarray) -> Optional[Dict]:
    """Score and select the best candidate contour for STOP sign."""
    best = None
    img_w, img_h = img_wh

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        hull = cv2.convexHull(c)
        hull_area = max(cv2.contourArea(hull), 1.0)
        solidity = float(area) / hull_area

        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h) if h > 0 else 0
        circularity = 4.0 * np.pi * area / (peri * peri + 1e-6)

        # geometry filters
        poly_ok = 7 <= len(approx) <= 10
        aspect_ok = 0.75 <= aspect <= 1.25
        solidity_ok = solidity > 0.85
        circ_ok = 0.55 <= circularity <= 0.90

        # red ratio in bbox
        roi = red_mask[y:y+h, x:x+w]
        red_ratio = float(cv2.countNonZero(roi)) / (w * h + 1e-6)
        red_ok = 0.10 <= red_ratio <= 0.75

        score = 0.0
        if poly_ok: score += 1.0
        if aspect_ok: score += 0.7
        if solidity_ok: score += 0.5
        if circ_ok: score += 0.5
        if red_ok: score += 0.8
        else: score -= 0.5

        score += min(area / (img_w * img_h), 0.5)  # area bonus

        if best is None or score > best["score"]:
            best = {
                "contour": c, "approx": approx, "bbox": (x, y, w, h),
                "area": area, "solidity": solidity, "circularity": circularity,
                "score": score, "red_ratio": red_ratio
            }
    return best


def detect_stop_in_image(bgr: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]], Dict]:
    """Run detection on one image."""
    work, scale = resize_max_dim(bgr, 1200)
    mask = red_mask_hsv(work)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = work.shape[:2]
    min_area = int(0.0005 * w * h)
    best = select_best_candidate(contours, min_area, (w, h), mask)

    if best is None:
        return False, None, {"mask": mask, "scale": scale, "work": work}

    x, y, bw, bh = best["bbox"]
    cx, cy = x + bw // 2, y + bh // 2
    return True, (x, y, bw, bh), {
        "mask": mask, "scale": scale, "center": (cx, cy),
        "score": best["score"], "work": work
    }


def annotate_and_save(work_img: np.ndarray, bbox: Tuple[int, int, int, int],
                      center: Tuple[int, int], out_path: str):
    x, y, w, h = bbox
    vis = work_img.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.drawMarker(vis, center, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.putText(vis, f"STOP cx={center[0]}, cy={center[1]}",
                (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Image folder (stop_sign_dataset)")
    ap.add_argument("--out", type=str, default="./results", help="Output folder")
    ap.add_argument("--save-mask", action="store_true", help="Save mask images too")
    ap.add_argument("--exts", type=str, default="jpg,jpeg,png,bmp", help="Extensions: jpg,jpeg,png,bmp")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # collect files
    patterns = [f"*.{e.strip()}" for e in args.exts.split(",")]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(args.data, p)))
    if not files:
        print("Warning: no matching images found in dataset folder.")
        return

    csv_path = os.path.join(args.out, "detections.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["filename", "found", "cx", "cy", "w", "h", "score", "red_ratio"])

        total, found_count = 0, 0
        for fp in sorted(files):
            total += 1
            img = cv2.imread(fp)
            if img is None:
                print(f"Could not read: {fp}")
                writer.writerow([os.path.basename(fp), 0, "", "", "", "", 0, ""])
                continue

            ok, bbox, dbg = detect_stop_in_image(img)
            if ok and bbox is not None:
                x, y, w, h = bbox
                cx, cy = dbg["center"]
                found_count += 1
                print(f"{os.path.basename(fp)} -> CENTER (cx={cx}, cy={cy})")
                out_img = os.path.join(args.out, os.path.splitext(os.path.basename(fp))[0] + "_annotated.jpg")
                annotate_and_save(dbg["work"], bbox, (cx, cy), out_img)
                writer.writerow([os.path.basename(fp), 1, cx, cy, w, h,
                                 round(dbg.get('score', 0.0), 3),
                                 round(dbg.get('red_ratio', 0.0), 3)])
            else:
                print(f"{os.path.basename(fp)} -> NO DETECTION")
                writer.writerow([os.path.basename(fp), 0, "", "", "", "", 0, ""])

            if args.save_mask and "mask" in dbg:
                mask_out = os.path.join(args.out, os.path.splitext(os.path.basename(fp))[0] + "_mask.png")
                cv2.imwrite(mask_out, dbg["mask"])

    print(f"\nDone. {total} images, {found_count} detections.")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

