#!/usr/bin/env python3
"""
Multi-component extraction for images 1-5.
Detects multiple components per image and saves each as a separate crop.

Usage:
    python3 src/extract_first5_components.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Try Pillow as fallback for corrupted JPEGs
try:
    from PIL import Image
except ImportError:
    Image = None


# ==================== CONFIGURATION ====================
INPUT_DIR = Path("Compoent_Images")
OUTPUT_DIR = Path("datasets/first5_multi")

IMAGE_IDS = [1, 2, 3, 4, 5]  # First 5 images
CLASS_NAME = "brake_caliper"  # Adjust based on actual component
CLASS_ID = 0

# Detection parameters (tune these if needed)
MIN_AREA_RATIO = 0.012      # Minimum component size (1.2% of image)
MAX_AREA_RATIO = 0.75       # Maximum component size (75% of image)
MAX_COMPONENTS = 10         # Max components to extract per image
PADDING = 25                # Padding around detected component

# Edge detection parameters
CANNY_LOW = 50
CANNY_HIGH = 150
MORPH_KERNEL_SIZE = 9

# Output options
SAVE_DEBUG = True           # Save debug images with boxes drawn
COPY_SOURCE = True          # Copy original images to output
# =======================================================


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG"}


@dataclass
class ComponentBox:
    """Represents a detected component bounding box."""
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


def ensure_output_structure(base: Path) -> dict[str, Path]:
    """Create output folder structure."""
    paths = {
        'base': base,
        'images': base / 'images',
        'crops': base / 'crops',
        'debug': base / 'debug',
        'manifests': base / 'manifests',
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Create classes.txt
    (base / 'classes.txt').write_text(f"{CLASS_NAME}\n", encoding='utf-8')
    
    return paths


def find_image_path(img_id: int) -> Optional[Path]:
    """Find image file with any extension (1.jpeg, 1.jpg, 1.png, etc)."""
    for ext in IMG_EXTENSIONS:
        path = INPUT_DIR / f"{img_id}{ext}"
        if path.exists():
            return path
    
    # Fallback: search with glob
    candidates = list(INPUT_DIR.glob(f"{img_id}.*"))
    for candidate in candidates:
        if candidate.suffix.lower() in {e.lower() for e in IMG_EXTENSIONS}:
            return candidate
    
    return None


def read_image_robust(path: Path) -> Optional[np.ndarray]:
    """Read image with fallback to PIL if cv2 fails."""
    # Try OpenCV first
    img = cv2.imread(str(path))
    if img is not None:
        return img
    
    # Fallback to PIL
    if Image is None:
        return None
    
    try:
        pil_img = Image.open(path).convert('RGB')
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"   ⚠️  PIL also failed: {e}")
        return None


def clamp(value: int, min_val: int, max_val: int) -> int:
    """Clamp value between min and max."""
    return max(min_val, min(value, max_val))


def detect_components(img: np.ndarray) -> List[ComponentBox]:
    """
    Detect multiple component candidates in the image.
    Uses edge detection + morphology + contour filtering.
    """
    h, w = img.shape[:2]
    img_area = w * h
    
    # 1) Convert to grayscale and smooth
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2) Edge detection
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    
    # 3) Morphological operations to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4) Find contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5) Filter and create bounding boxes
    boxes: List[ComponentBox] = []
    
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        
        # Filter by area ratio
        area_ratio = area / img_area
        if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
            continue
        
        # Filter out very thin/tall boxes (likely noise)
        aspect_ratio = bw / bh if bh > 0 else 0
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            continue
        
        # Filter out very small dimensions
        if bw < 30 or bh < 30:
            continue
        
        boxes.append(ComponentBox(x, y, x + bw, y + bh, area))
    
    # 6) Sort by area (largest first) and keep top N
    boxes.sort(key=lambda b: b.area, reverse=True)
    boxes = boxes[:MAX_COMPONENTS]
    
    # 7) Apply padding and clamp to image bounds
    padded_boxes: List[ComponentBox] = []
    for box in boxes:
        x1 = clamp(box.x1 - PADDING, 0, w - 1)
        y1 = clamp(box.y1 - PADDING, 0, h - 1)
        x2 = clamp(box.x2 + PADDING, 0, w - 1)
        y2 = clamp(box.y2 + PADDING, 0, h - 1)
        padded_boxes.append(ComponentBox(x1, y1, x2, y2, (x2-x1)*(y2-y1)))
    
    return padded_boxes


def process_image(img_id: int, paths: dict[str, Path]) -> Tuple[int, List[dict]]:
    """Process one image and return number of components found."""
    img_path = find_image_path(img_id)
    
    if img_path is None:
        print(f"[{img_id}] ❌ Image file not found (tried: {img_id}.jpeg, {img_id}.jpg, etc)")
        return 0, []
    
    # Read image
    img = read_image_robust(img_path)
    if img is None:
        print(f"[{img_id}] ❌ Cannot read image: {img_path.name}")
        return 0, []
    
    # Detect components
    boxes = detect_components(img)
    
    if not boxes:
        print(f"[{img_id}] ⚠️  No components detected (try adjusting MIN_AREA_RATIO)")
        return 0, []
    
    h, w = img.shape[:2]
    manifest_rows = []
    
    # Draw debug image if enabled
    if SAVE_DEBUG:
        debug_img = img.copy()
    
    # Save each component crop
    for idx, box in enumerate(boxes, start=1):
        # Extract crop
        crop = img[box.y1:box.y2, box.x1:box.x2]
        
        # Save crop
        crop_filename = f"{img_id}_{idx:03d}.jpg"
        crop_path = paths['crops'] / crop_filename
        cv2.imwrite(str(crop_path), crop)
        
        # Add to manifest
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
        
        # Draw on debug image
        if SAVE_DEBUG:
            # Green box for the component
            cv2.rectangle(debug_img, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 3)
            # Label
            label = f"{idx}"
            cv2.putText(
                debug_img, label,
                (box.x1, max(30, box.y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 2, cv2.LINE_AA
            )
    
    # Save debug image
    if SAVE_DEBUG:
        debug_path = paths['debug'] / f"{img_id}_components.jpg"
        cv2.imwrite(str(debug_path), debug_img)
    
    # Copy source image if enabled
    if COPY_SOURCE:
        import shutil
        dest = paths['images'] / img_path.name
        shutil.copy2(img_path, dest)
    
    print(f"[{img_id}] ✅ Found {len(boxes)} component(s) → saved {len(manifest_rows)} crops")
    
    return len(boxes), manifest_rows


def main():
    """Main execution function."""
    print(f"\n{'='*60}")
    print("🔍 Multi-Component Extraction for First 5 Images")
    print(f"{'='*60}\n")
    
    print(f"📂 Input:  {INPUT_DIR.resolve()}")
    print(f"📂 Output: {OUTPUT_DIR.resolve()}")
    print(f"🎯 Processing images: {IMAGE_IDS}\n")
    
    # Setup output structure
    paths = ensure_output_structure(OUTPUT_DIR)
    
    # Process each image
    total_components = 0
    all_manifest_rows = []
    
    for img_id in IMAGE_IDS:
        num_components, rows = process_image(img_id, paths)
        total_components += num_components
        all_manifest_rows.extend(rows)
    
    # Write manifest CSV
    if all_manifest_rows:
        manifest_path = paths['manifests'] / 'manifest_first5.csv'
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'crop_filename', 'crop_path', 'source_image', 'img_id',
                'component_idx', 'x1', 'y1', 'x2', 'y2', 'width', 'height'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_manifest_rows)
        
        print(f"\n📄 Manifest saved: {manifest_path.resolve()}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE")
    print(f"{'='*60}")
    print(f"Total component crops extracted: {len(all_manifest_rows)}")
    print(f"\n📁 Check outputs:")
    print(f"   • Crops:  {paths['crops']}")
    print(f"   • Debug:  {paths['debug']}")
    print(f"   • Images: {paths['images']}")
    print(f"\n🎯 Next: Run edge/grabcut pipelines on crops folder")


if __name__ == "__main__":
    main()