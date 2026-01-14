#!/usr/bin/env python3
"""
Batch processing using GrabCut segmentation for component extraction.
Processes a folder of cropped images and generates YOLO labels.

Usage:
    python3 src/build_batch_grabcut.py --in <input_folder> --class <class_name> --batch-name <output_name>
"""

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Batch GrabCut processing")
    parser.add_argument("--in", dest="input_dir", required=True, help="Input directory with images")
    parser.add_argument("--class", dest="class_name", required=True, help="Class name for detection")
    parser.add_argument("--batch-name", required=True, help="Output batch folder name")
    parser.add_argument("--iters", type=int, default=5, help="GrabCut iterations (default: 5)")
    parser.add_argument("--padding", type=int, default=10, help="Padding around crop (default: 10)")
    parser.add_argument("--save-debug", action="store_true", help="Save debug images")
    return parser.parse_args()


def ensure_output_structure(base: Path) -> dict:
    """Create output folder structure."""
    paths = {
        'base': base,
        'images': base / 'images',
        'labels': base / 'labels',
        'crops': base / 'crops',
        'debug': base / 'debug',
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths


def process_image_grabcut(
    img_path: Path,
    class_name: str,
    class_id: int,
    iters: int,
    padding: int,
    save_debug: bool,
    paths: dict
) -> Optional[dict]:
    """Process one image using GrabCut."""
    
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Initialize GrabCut with central rectangle
    margin = 0.05
    rect = (
        int(w * margin),
        int(h * margin),
        int(w * (1 - 2 * margin)),
        int(h * (1 - 2 * margin))
    )
    
    # Run GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_RECT)
    except:
        return None
    
    # Extract foreground
    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find largest foreground contour
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    
    # Apply padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)
    
    # YOLO format (normalized)
    x_center = (x1 + x2) / 2 / w
    y_center = (y1 + y2) / 2 / h
    bbox_w = (x2 - x1) / w
    bbox_h = (y2 - y1) / h
    
    # Save outputs
    img_name = img_path.stem
    
    # Save label
    label_path = paths['labels'] / f"{img_name}.txt"
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")
    
    # Apply mask and save crop
    img_masked = img * mask2[:, :, np.newaxis]
    crop = img_masked[y1:y2, x1:x2]
    crop_path = paths['crops'] / f"{img_name}_{class_name}.jpg"
    cv2.imwrite(str(crop_path), crop)
    
    # Save debug image
    if save_debug:
        debug_img = img.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Overlay mask
        debug_img[mask2 == 0] = debug_img[mask2 == 0] * 0.3
        debug_path = paths['debug'] / f"{img_name}_grabcut.jpg"
        cv2.imwrite(str(debug_path), debug_img)
    
    # Copy original image
    import shutil
    dest_img = paths['images'] / img_path.name
    shutil.copy2(img_path, dest_img)
    
    return {
        'crop_filename': crop_path.name,
        'crop_path': str(crop_path.resolve()),
        'source_image': str(img_path.resolve()),
        'class_name': class_name,
        'class_id': class_id,
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
    }


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path("outputs") / args.batch_name
    
    print(f"\n{'='*70}")
    print(f"🔍 BATCH GRABCUT PROCESSING")
    print(f"{'='*70}\n")
    print(f"📂 Input:  {input_dir.resolve()}")
    print(f"📂 Output: {output_dir.resolve()}")
    print(f"🏷️  Class:  {args.class_name}")
    print(f"🔄 GrabCut iterations: {args.iters}")
    print(f"📦 Padding: {args.padding}px\n")
    
    # Setup output structure
    paths = ensure_output_structure(output_dir)
    
    # Write classes.txt
    (output_dir / 'classes.txt').write_text(f"{args.class_name}\n")
    
    # Get all images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']:
        image_files.extend(input_dir.glob(f"*{ext}"))
    
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} images to process\n")
    
    # Process images
    results = []
    processed = 0
    failed = 0
    
    for i, img_path in enumerate(image_files, 1):
        result = process_image_grabcut(
            img_path,
            args.class_name,
            0,  # class_id
            args.iters,
            args.padding,
            args.save_debug,
            paths
        )
        
        if result:
            results.append(result)
            processed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"
        
        if i % 50 == 0 or i == len(image_files):
            print(f"[{i}/{len(image_files)}] {status} {img_path.name}")
    
    # Write CSV manifest
    if results:
        csv_path = output_dir / 'crops_labels.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n📄 Manifest: {csv_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE")
    print(f"{'='*70}")
    print(f"Processed: {processed}/{len(image_files)}")
    print(f"Failed:    {failed}")
    print(f"\n📁 Outputs:")
    print(f"   • Labels: {paths['labels']}")
    print(f"   • Crops:  {paths['crops']}")
    if args.save_debug:
        print(f"   • Debug:  {paths['debug']}")


if __name__ == "__main__":
    main()