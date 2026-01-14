#!/usr/bin/env python3
"""
FIRST5 ONLY (Edges baseline)

Input:  Compoent_Images/{1..5}.jpeg
Output: outputs/first5_edges/{images,labels,crops,debug,classes.txt,crops_labels.csv}

Guaranteed:
- processes ONLY 1.jpeg..5.jpeg
- creates ONLY labels 1.txt..5.txt
"""

import csv
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "Compoent_Images"
OUT_BASE = ROOT / "outputs" / "first5_edges"

CLASS_NAME = "brake_caliper"
CLASS_ID = 0

PADDING_PX = 20
COPY_IMAGES = True
SAVE_DEBUG = True


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


def find_bbox_edges(img, min_area_ratio=0.01, max_area_ratio=0.75):
    h, w = img.shape[:2]
    img_area = w * h

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)
    img2 = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    edges = cv2.Canny(gray, 40, 140)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.dilate(edges, kernel, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, edges

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:8]:
        area = cv2.contourArea(c)
        if min_area_ratio * img_area <= area <= max_area_ratio * img_area:
            x, y, bw, bh = cv2.boundingRect(c)
            return (x, y, x + bw, y + bh), edges

    return None, edges


def main():
    ensure_dirs()

    selected = [INPUT_DIR / f"{i}.jpeg" for i in range(1, 6)]
    missing = [p.name for p in selected if not p.exists()]
    if missing:
        raise SystemExit(f"Lipsesc din Compoent_Images: {missing}")

    rows = []
    ok = 0

    for i, img_path in enumerate(selected, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            raise SystemExit(f"Nu pot citi: {img_path}")

        bbox, edges = find_bbox_edges(img)
        if bbox is None:
            raise SystemExit(f"Nu am gasit bbox (edges) pentru: {img_path.name}")

        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox

        xc, yc, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)

        # IMPORTANT: label name forced to i.txt (1..5)
        (OUT_BASE / "labels" / f"{i}.txt").write_text(
            f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n",
            encoding="utf-8",
        )

        # crop
        x1p = clamp(x1 - PADDING_PX, 0, w - 1)
        y1p = clamp(y1 - PADDING_PX, 0, h - 1)
        x2p = clamp(x2 + PADDING_PX, 0, w - 1)
        y2p = clamp(y2 + PADDING_PX, 0, h - 1)
        crop = img[y1p:y2p, x1p:x2p]
        crop_path = OUT_BASE / "crops" / f"{i}_{CLASS_NAME}.jpg"
        cv2.imwrite(str(crop_path), crop)

        if COPY_IMAGES:
            shutil.copy2(img_path, OUT_BASE / "images" / f"{i}.jpeg")

        if SAVE_DEBUG:
            dbg = img.copy()
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(dbg, CLASS_NAME, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(str(OUT_BASE / "debug" / f"{i}_bbox.jpg"), dbg)
            cv2.imwrite(str(OUT_BASE / "debug" / f"{i}_edges.jpg"), edges)

        rows.append([str(crop_path), CLASS_NAME, str(img_path), CLASS_ID, x1, y1, x2, y2])
        ok += 1
        print(f"[{ok}/5] {img_path.name} ✅")

    # CSV
    csv_path = OUT_BASE / "crops_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_path", "class_name", "source_image", "class_id", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)

    print(f"\nDone (edges). Output: {OUT_BASE}")
    print(f"Labels: {OUT_BASE / 'labels'}")


if __name__ == "__main__":
    main()
