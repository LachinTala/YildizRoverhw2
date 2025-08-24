#!/usr/bin/env python3
"""
Simple evaluation tool.
- Reads detections.csv (from detect_stop.py).
- Optionally compares with ground truth CSV (filename, gt_x, gt_y, gt_w, gt_h).
"""

import argparse
import csv
from typing import Dict, Tuple


def iou(boxA, boxB) -> float:
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    xA = max(ax, bx)
    yA = max(ay, by)
    xB = min(ax + aw, bx + bw)
    yB = min(ay + ah, by + bh)
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union


def read_pred(path: str) -> Dict[str, Tuple[int, Tuple[int, int, int, int]]]:
    res = {}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            fn = r["filename"]
            found = int(r["found"]) if r["found"] else 0
            if found:
                try:
                    cx = int(float(r["cx"])); cy = int(float(r["cy"]))
                    w = int(float(r["w"])); h = int(float(r["h"]))
                    # currently we don't store x,y in CSV, so IoU can't be computed
                    res[fn] = (1, None)
                except:
                    res[fn] = (1, None)
            else:
                res[fn] = (0, None)
    return res


def read_gt(path: str) -> Dict[str, Tuple[int, int, int, int]]:
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            fn = r["filename"]
            gt[fn] = (int(r["gt_x"]), int(r["gt_y"]), int(r["gt_w"]), int(r["gt_h"]))
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="detections.csv path")
    ap.add_argument("--gt", help="Ground truth CSV (optional)")
    args = ap.parse_args()

    preds = read_pred(args.pred)

    total = len(preds)
    detected = sum(1 for v in preds.values() if v[0] == 1)
    print(f"Total: {total} images, Detected: {detected}  (Detection Rate: {detected/total:.3f})")

    if args.gt:
        print("\n[Note] Current detections.csv does not store x,y, so IoU cannot be computed directly.")
        print("To compute IoU, modify detect_stop.py to also save x,y coordinates.")


if __name__ == "__main__":
    main()
