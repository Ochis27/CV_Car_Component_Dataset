#!/usr/bin/env python3
"""
Multi-component extraction for ALL images (1-275).
Optimized for batch processing with progress tracking.

Usage:
    python3 src/extract_all_components.py
    python3 src/extract_all_components.py --start 1 --end 100  # Process subset
    python3 src/extract_all_components.py --resume  # Skip already processed
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


# ==================== CONFIGURATION ====================
INPUT_DIR = Path("Compoent_Images")
OUTPUT_DIR = Path("datasets/all_components")

CLASS_NAME = "brake_caliper"
CLASS_ID = 0

# Detection parameters
MIN_AREA_RATIO = 0.012
MAX_AREA_RATIO = 0.75
MAX_COMPONENTS = 10
PADDING = 25

# Edge detection
CANNY_LOW = 50
CANNY_HIGH = 150
MORPH_KERNEL_SIZE = 9

# Output options
SAVE_DEBUG = True
COPY_SOURCE = False  # Set to False for 275 images to save space
# =======================================================


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"}


@dataclass
class ComponentBox:
    x1: int
    y1: int
    x2: int
    y2: int
    area: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1


def parse_args():
    parser = argparse.ArgumentParser(description="Extract components from all images")
    parser.add_argument("--start", type=int, default=1, help="Start image ID")
    parser.add_argument("--end", type=int, default=275, help="End image ID")
    parser.add_argument("--resume", action="store_true", help="Skip already processed images")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    return parser.parse_args()


def ensure_output_structure(base: Path) -> dict[str, Path]:
    paths = {
        'base': base,
        'images': base / 'images',
        'crops': base / 'crops',
        'debug': base / 'debug',
        'manifests': base / 'manifests',
        'logs': base / 'logs',
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    (base / 'classes.txt').write_text(f"{CLASS_NAME}\n", encoding='utf-8')
    
    return paths


def find_image_path(img_id: int) -> Optional[Path]:
    for ext in IMG_EXTENSIONS:
        path = INPUT_DIR / f"{img_id}{ext}"
        if path.exists():
            return path
    
    candidates = list(INPUT_DIR.glob(f"{img_id}.*"))
    for candidate in candidates:
        if candidate.suffix.lower() in {e.lower() for e in IMG_EXTENSIONS}:
            return candidate
    
    return None


def read_image_robust(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is not None:
        return img
    
    if Image is None:
        return None
    
    try:
        pil_img = Image.open(path).convert('RGB')
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def clamp(value: int, min_val: int, max_val: int) -> int:
    return max(min_val, min(value, max_val))


def detect_components(img: np.ndarray) -> List[ComponentBox]:
    h, w = img.shape[:2]
    img_area = w * h
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes: List[ComponentBox] = []
    
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        
        area_ratio = area / img_area
        if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
            continue
        
        aspect_ratio = bw / bh if bh > 0 else 0
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            continue
        
        if bw < 30 or bh < 30:
            continue
        
        boxes.append(ComponentBox(x, y, x + bw, y + bh, area))
    
    boxes.sort(key=lambda b: b.area, reverse=True)
    boxes = boxes[:MAX_COMPONENTS]
    
    padded_boxes: List[ComponentBox] = []
    for box in boxes:
        x1 = clamp(box.x1 - PADDING, 0, w - 1)
        y1 = clamp(box.y1 - PADDING, 0, h - 1)
        x2 = clamp(box.x2 + PADDING, 0, w - 1)
        y2 = clamp(box.y2 + PADDING, 0, h - 1)
        padded_boxes.append(ComponentBox(x1, y1, x2, y2, (x2-x1)*(y2-y1)))
    
    return padded_boxes


def is_already_processed(img_id: int, paths: dict[str, Path]) -> bool:
    """Check if image was already processed by looking for crops."""
    pattern = f"{img_id}_*.jpg"
    existing = list(paths['crops'].glob(pattern))
    return len(existing) > 0


def process_image(
    img_id: int, 
    paths: dict[str, Path], 
    save_debug: bool = True
) -> Tuple[int, List[dict], Optional[str]]:
    """
    Process one image and return (num_components, manifest_rows, error_msg).
    """
    img_path = find_image_path(img_id)
    
    if img_path is None:
        return 0, [], "not_found"
    
    img = read_image_robust(img_path)
    if img is None:
        return 0, [], "cannot_read"
    
    boxes = detect_components(img)
    
    if not boxes:
        return 0, [], "no_components"
    
    h, w = img.shape[:2]
    manifest_rows = []
    
    if save_debug:
        debug_img = img.copy()
    
    for idx, box in enumerate(boxes, start=1):
        crop = img[box.y1:box.y2, box.x1:box.x2]
        
        crop_filename = f"{img_id}_{idx:03d}.jpg"
        crop_path = paths['crops'] / crop_filename
        cv2.imwrite(str(crop_path), crop)
        
        manifest_rows.append({
            'crop_filename': crop_filename,
            'crop_path': str(crop_path.resolve()),
            'source_image': str(img_path.resolve()),
            'img_id': img_id,
            'component_idx': idx,
            'x1': box.x1,
            'y1': box.y1,
            'x2': box.x2,
            'y2': box.y2,
            'width': box.width,
            'height': box.height,
        })
        
        if save_debug:
            cv2.rectangle(debug_img, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 3)
            label = f"{idx}"
            cv2.putText(
                debug_img, label,
                (box.x1, max(30, box.y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 2, cv2.LINE_AA
            )
    
    if save_debug:
        debug_path = paths['debug'] / f"{img_id}_components.jpg"
        cv2.imwrite(str(debug_path), debug_img)
    
    if COPY_SOURCE:
        import shutil
        dest = paths['images'] / img_path.name
        shutil.copy2(img_path, dest)
    
    return len(boxes), manifest_rows, None


def print_progress(current: int, total: int, start_time: float, status: str = ""):
    """Print progress bar with time estimate."""
    percent = (current / total) * 100
    elapsed = time.time() - start_time
    
    if current > 0:
        avg_time = elapsed / current
        remaining = avg_time * (total - current)
        eta_min = int(remaining / 60)
        eta_sec = int(remaining % 60)
        eta = f"{eta_min}m {eta_sec}s"
    else:
        eta = "calculating..."
    
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    print(f"\r[{bar}] {current}/{total} ({percent:.1f}%) | ETA: {eta} | {status}", end="", flush=True)


def main():
    args = parse_args()
    
    print(f"\n{'='*70}")
    print("🔍 BATCH COMPONENT EXTRACTION - ALL IMAGES")
    print(f"{'='*70}\n")
    
    print(f"📂 Input:  {INPUT_DIR.resolve()}")
    print(f"📂 Output: {OUTPUT_DIR.resolve()}")
    print(f"🎯 Processing images: {args.start} → {args.end}")
    if args.resume:
        print(f"🔄 Resume mode: skipping already processed images")
    print()
    
    paths = ensure_output_structure(OUTPUT_DIR)
    
    all_manifest_rows = []
    stats = {
        'processed': 0,
        'skipped': 0,
        'not_found': 0,
        'cannot_read': 0,
        'no_components': 0,
        'total_crops': 0,
    }
    
    failed_images = []
    start_time = time.time()
    
    total_images = args.end - args.start + 1
    
    for i, img_id in enumerate(range(args.start, args.end + 1), start=1):
        # Check if already processed
        if args.resume and is_already_processed(img_id, paths):
            stats['skipped'] += 1
            print_progress(i, total_images, start_time, f"Skipped {img_id} (already done)")
            continue
        
        # Process image
        num_comps, rows, error = process_image(img_id, paths, save_debug=args.debug or SAVE_DEBUG)
        
        if error:
            stats[error] += 1
            failed_images.append((img_id, error))
            status = f"⚠️ {img_id}: {error}"
        else:
            stats['processed'] += 1
            stats['total_crops'] += num_comps
            all_manifest_rows.extend(rows)
            status = f"✅ {img_id}: {num_comps} component(s)"
        
        print_progress(i, total_images, start_time, status)
    
    print("\n")  # New line after progress bar
    
    # Write manifest
    if all_manifest_rows:
        manifest_path = paths['manifests'] / 'manifest_all.csv'
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'crop_filename', 'crop_path', 'source_image', 'img_id',
                'component_idx', 'x1', 'y1', 'x2', 'y2', 'width', 'height'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_manifest_rows)
        
        print(f"📄 Manifest: {manifest_path.resolve()}")
    
    # Write failed images log
    if failed_images:
        log_path = paths['logs'] / 'failed_images.txt'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("Image ID | Error\n")
            f.write("---------|-------\n")
            for img_id, error in failed_images:
                f.write(f"{img_id:8d} | {error}\n")
        print(f"⚠️  Failed: {log_path.resolve()}")
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE - Took {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    print(f"Processed:      {stats['processed']}")
    print(f"Skipped:        {stats['skipped']}")
    print(f"Not found:      {stats['not_found']}")
    print(f"Cannot read:    {stats['cannot_read']}")
    print(f"No components:  {stats['no_components']}")
    print(f"Total crops:    {stats['total_crops']}")
    print(f"\n📁 Outputs:")
    print(f"   • Crops:  {paths['crops']}")
    if args.debug or SAVE_DEBUG:
        print(f"   • Debug:  {paths['debug']}")
    print(f"\n🎯 Next: Run edge/grabcut pipelines on {paths['crops']}")


if __name__ == "__main__":
    main()