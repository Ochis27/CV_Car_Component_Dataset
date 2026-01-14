#!/usr/bin/env python3
"""
FIRST5 ONLY (GrabCut improved)

Input:  Compoent_Images/{1..5}.jpeg
Output: outputs/first5_grabcut/{images,labels,crops,debug,classes.txt,crops_labels.csv}

Guaranteed:
- processes ONLY 1.jpeg..5.jpeg
- creates ONLY labels 1.txt..5.txt
"""

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "Compoent_Images"
OUT_BASE = ROOT / "outputs" / "first5_grabcut"

CLASS_NAME = "brake_caliper"
CLASS_ID = 0

PADDING_PX = 30
COPY_IMAGES = True
SAVE_DEBUG = True

MAX_DIM = 900
GRABCUT_ITERS = 3


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    return xc, yc, bw, bh


def ensure_dirs():
    (OUT_BASE / "images").mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "labels").mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "crops").mkdir(parents=True, exist_ok=True)
    if SAVE_DEBUG:
        (OUT_BASE / "debug").mkdir(parents=True, exist_ok=True)

    (OUT_BASE / "classes.txt").write_text(CLASS_NAME + "\n", encoding="utf-8")


def resize_keep_aspect(img, max_dim):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def grabcut_bbox(img):
    h, w = img.shape[:2]

    rx = int(0.12 * w)
    ry = int(0.12 * h)
    rw = int(0.76 * w)
    rh = int(0.76 * h)
    rect = (rx, ry, rw, rh)

    mask = np.zeros((h, w), np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgModel, fgModel, GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, fg_mask

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    x, y, bw, bh = cv2.boundingRect(cnts[0])

    if (bw * bh) < 0.02 * (w * h):
        return None, fg_mask

    return (x, y, x + bw, y + bh), fg_mask


def main():
    ensure_dirs()

    selected = [INPUT_DIR / f"{i}.jpeg" for i in range(1, 6)]
    missing = [p.name for p in selected if not p.exists()]
    if missing:
        raise SystemExit(f"Lipsesc din Compoent_Images: {missing}")

    rows = []
    ok = 0

    for i, img_path in enumerate(selected, start=1):
        img_full = cv2.imread(str(img_path))
        if img_full is None:
            raise SystemExit(f"Nu pot citi: {img_path}")

        img_small, scale = resize_keep_aspect(img_full, MAX_DIM)
        bbox_small, fg_mask_small = grabcut_bbox(img_small)

        if bbox_small is None:
            raise SystemExit(f"Nu am gasit bbox (grabcut) pentru: {img_path.name}")

        x1, y1, x2, y2 = bbox_small
        if scale != 1.0:
            x1 = int(x1 / scale); y1 = int(y1 / scale)
            x2 = int(x2 / scale); y2 = int(y2 / scale)

        h, w = img_full.shape[:2]
        x1 = clamp(x1, 0, w - 1); x2 = clamp(x2, 0, w - 1)
        y1 = clamp(y1, 0, h - 1); y2 = clamp(y2, 0, h - 1)

        xc, yc, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)

        # IMPORTANT: label name forced to i.txt (1..5)
        (OUT_BASE / "labels" / f"{i}.txt").write_text(
            f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n",
            encoding="utf-8",
        )

        # crop with padding
        x1p = clamp(x1 - PADDING_PX, 0, w - 1)
        y1p = clamp(y1 - PADDING_PX, 0, h - 1)
        x2p = clamp(x2 + PADDING_PX, 0, w - 1)
        y2p = clamp(y2 + PADDING_PX, 0, h - 1)
        crop = img_full[y1p:y2p, x1p:x2p]
        crop_path = OUT_BASE / "crops" / f"{i}_{CLASS_NAME}.jpg"
        cv2.imwrite(str(crop_path), crop)

        if COPY_IMAGES:
            shutil.copy2(img_path, OUT_BASE / "images" / f"{i}.jpeg")

        if SAVE_DEBUG:
            dbg = img_full.copy()
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(dbg, CLASS_NAME, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(str(OUT_BASE / "debug" / f"{i}_bbox.jpg"), dbg)

            if scale != 1.0:
                fg_mask = cv2.resize(fg_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                fg_mask = fg_mask_small
            cv2.imwrite(str(OUT_BASE / "debug" / f"{i}_fgmask.jpg"), fg_mask)

        rows.append([str(crop_path), CLASS_NAME, str(img_path), CLASS_ID, x1, y1, x2, y2])
        ok += 1
        print(f"[{ok}/5] {img_path.name} ✅")

    csv_path = OUT_BASE / "crops_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_path", "class_name", "source_image", "class_id", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)

    print(f"\nDone (grabcut). Output: {OUT_BASE}")
    print(f"Labels: {OUT_BASE / 'labels'}")


if __name__ == "__main__":
    main()
