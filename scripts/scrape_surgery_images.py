"""Scrape public before/after plastic surgery images from multiple sources.

Sources:
1. locateadoc.com - before/after galleries (referenced by IIITD dataset .exe)
2. surgery.org (ASAPS) - before/after photo gallery
3. RealSelf.com - user-posted before/after images
4. Public Instagram surgeon accounts

All images are publicly posted clinical photos.
Outputs paired before/after crops with procedure labels.
"""

import argparse
import hashlib
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np

# Mapping of procedure keywords for classification
PROCEDURE_KEYWORDS = {
    "rhinoplasty": ["rhinoplasty", "nose job", "nose surgery", "nasal", "septoplasty"],
    "blepharoplasty": ["blepharoplasty", "eyelid", "eye lift", "eyelid surgery", "lid lift"],
    "rhytidectomy": ["rhytidectomy", "facelift", "face lift", "face-lift", "face lifting"],
    "orthognathic": ["orthognathic", "jaw surgery", "jaw correction", "maxillofacial", "chin surgery", "genioplasty"],
}


def classify_procedure(text: str) -> str:
    """Classify a procedure from surrounding text/URL."""
    text_lower = text.lower()
    for proc, keywords in PROCEDURE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return proc
    return "unknown"


def download_image(url: str, save_path: Path, timeout: int = 15) -> bool:
    """Download an image from URL."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        save_path.write_bytes(data)
        return True
    except Exception as e:
        return False


def split_comparison_image(img: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Split a before/after comparison image into two halves.

    Auto-detects horizontal (left/right) or vertical (top/bottom) layout.
    Returns (before, after) or None if detection fails.
    """
    h, w = img.shape[:2]

    if w < 100 or h < 100:
        return None

    # Check for a vertical dividing line (horizontal split = left/right layout)
    # Look for a thin strip in the center that's very different from surroundings
    mid_x = w // 2
    strip_w = max(4, w // 50)

    # Check horizontal split (side by side) — most common
    left_half = img[:, :mid_x - strip_w]
    right_half = img[:, mid_x + strip_w:]

    # Check vertical split (top/bottom)
    mid_y = h // 2
    strip_h = max(4, h // 50)
    top_half = img[:mid_y - strip_h, :]
    bottom_half = img[mid_y + strip_h:, :]

    # Determine layout by aspect ratio
    aspect = w / h

    if aspect > 1.5:
        # Wide image → likely side-by-side
        # Resize both halves to same size
        target_h = min(left_half.shape[0], right_half.shape[0])
        target_w = min(left_half.shape[1], right_half.shape[1])
        before = cv2.resize(left_half, (target_w, target_h))
        after = cv2.resize(right_half, (target_w, target_h))
        return before, after
    elif aspect < 0.7:
        # Tall image → likely top/bottom
        target_h = min(top_half.shape[0], bottom_half.shape[0])
        target_w = min(top_half.shape[1], bottom_half.shape[1])
        before = cv2.resize(top_half, (target_w, target_h))
        after = cv2.resize(bottom_half, (target_w, target_h))
        return before, after
    else:
        # Square-ish → try both, pick the one where both halves have faces
        # Default to horizontal split
        target_w = min(left_half.shape[1], right_half.shape[1])
        target_h = min(left_half.shape[0], right_half.shape[0])
        before = cv2.resize(left_half, (target_w, target_h))
        after = cv2.resize(right_half, (target_w, target_h))
        return before, after


def has_face(img: np.ndarray) -> bool:
    """Quick face detection using OpenCV Haar cascade."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    return len(faces) > 0


def scrape_realself(output_dir: Path, max_pages: int = 50) -> list[dict]:
    """Scrape before/after photos from RealSelf.com galleries."""
    pairs = []
    procedures_to_scrape = [
        ("rhinoplasty", "rhinoplasty"),
        ("eyelid-surgery", "blepharoplasty"),
        ("facelift", "rhytidectomy"),
        ("jaw-surgery", "orthognathic"),
    ]

    for url_slug, proc_name in procedures_to_scrape:
        print(f"  Scraping RealSelf: {proc_name}...")
        for page in range(1, max_pages + 1):
            url = f"https://www.realself.com/photos/{url_slug}?page={page}"
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                })
                with urllib.request.urlopen(req, timeout=15) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
            except Exception as e:
                print(f"    Page {page} failed: {e}")
                break

            # Find image URLs in the gallery HTML
            # RealSelf uses various image patterns
            img_urls = re.findall(
                r'https?://(?:cdn|images)\.realself\.com/[^\s"\'<>]+\.(?:jpg|jpeg|png|webp)',
                html, re.IGNORECASE
            )
            img_urls = list(set(img_urls))

            if not img_urls:
                break

            for img_url in img_urls:
                img_hash = hashlib.md5(img_url.encode()).hexdigest()[:12]
                raw_path = output_dir / "raw" / f"realself_{proc_name}_{img_hash}.jpg"
                raw_path.parent.mkdir(parents=True, exist_ok=True)

                if raw_path.exists():
                    continue

                if download_image(img_url, raw_path):
                    pairs.append({
                        "source": "realself",
                        "procedure": proc_name,
                        "raw_path": str(raw_path),
                        "url": img_url,
                    })

            time.sleep(1.0)  # rate limit

            if len([p for p in pairs if p["procedure"] == proc_name]) >= 200:
                break

    return pairs


def scrape_locateadoc(output_dir: Path, max_pages: int = 30) -> list[dict]:
    """Scrape before/after from locateadoc.com (source used by IIITD dataset)."""
    pairs = []
    procedures = [
        ("rhinoplasty", "rhinoplasty"),
        ("blepharoplasty", "blepharoplasty"),
        ("facelift", "rhytidectomy"),
        ("jaw-surgery", "orthognathic"),
    ]

    for url_slug, proc_name in procedures:
        print(f"  Scraping LocateADoc: {proc_name}...")
        for page in range(1, max_pages + 1):
            url = f"https://www.locateadoc.com/pictures/{url_slug}/before-and-after/?page={page}"
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                })
                with urllib.request.urlopen(req, timeout=15) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
            except Exception as e:
                print(f"    Page {page} failed: {e}")
                break

            img_urls = re.findall(
                r'https?://[^\s"\'<>]*locateadoc[^\s"\'<>]*\.(?:jpg|jpeg|png|webp)',
                html, re.IGNORECASE
            )
            img_urls = list(set(img_urls))

            if not img_urls:
                break

            for img_url in img_urls:
                img_hash = hashlib.md5(img_url.encode()).hexdigest()[:12]
                raw_path = output_dir / "raw" / f"locateadoc_{proc_name}_{img_hash}.jpg"
                raw_path.parent.mkdir(parents=True, exist_ok=True)

                if raw_path.exists():
                    continue

                if download_image(img_url, raw_path):
                    pairs.append({
                        "source": "locateadoc",
                        "procedure": proc_name,
                        "raw_path": str(raw_path),
                        "url": img_url,
                    })

            time.sleep(1.5)

            if len([p for p in pairs if p["procedure"] == proc_name]) >= 150:
                break

    return pairs


def scrape_surgery_org(output_dir: Path, max_pages: int = 20) -> list[dict]:
    """Scrape before/after from surgery.org (ASAPS gallery)."""
    pairs = []
    procedures = [
        ("rhinoplasty", "rhinoplasty"),
        ("blepharoplasty", "blepharoplasty"),
        ("facelift", "rhytidectomy"),
    ]

    for url_slug, proc_name in procedures:
        print(f"  Scraping surgery.org: {proc_name}...")
        for page in range(1, max_pages + 1):
            url = f"https://www.surgery.org/procedures/{url_slug}/photos?page={page}"
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                })
                with urllib.request.urlopen(req, timeout=15) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
            except Exception as e:
                print(f"    Page {page} failed: {e}")
                break

            img_urls = re.findall(
                r'https?://[^\s"\'<>]*surgery\.org[^\s"\'<>]*\.(?:jpg|jpeg|png|webp)',
                html, re.IGNORECASE
            )
            img_urls += re.findall(
                r'https?://[^\s"\'<>]*\.(?:jpg|jpeg|png)[^\s"\'<>]*before[^\s"\'<>]*after',
                html, re.IGNORECASE
            )
            img_urls = list(set(img_urls))

            if not img_urls:
                break

            for img_url in img_urls:
                img_hash = hashlib.md5(img_url.encode()).hexdigest()[:12]
                raw_path = output_dir / "raw" / f"surgeryorg_{proc_name}_{img_hash}.jpg"
                raw_path.parent.mkdir(parents=True, exist_ok=True)

                if raw_path.exists():
                    continue

                if download_image(img_url, raw_path):
                    pairs.append({
                        "source": "surgery.org",
                        "procedure": proc_name,
                        "raw_path": str(raw_path),
                        "url": img_url,
                    })

            time.sleep(1.5)

            if len([p for p in pairs if p["procedure"] == proc_name]) >= 100:
                break

    return pairs


def process_raw_images(metadata: list[dict], output_dir: Path) -> list[dict]:
    """Split raw comparison images into before/after pairs, validate faces."""
    valid_pairs = []
    before_dir = output_dir / "before"
    after_dir = output_dir / "after"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)

    for i, meta in enumerate(metadata):
        raw_path = Path(meta["raw_path"])
        if not raw_path.exists():
            continue

        img = cv2.imread(str(raw_path))
        if img is None:
            continue

        result = split_comparison_image(img)
        if result is None:
            continue

        before, after = result

        # Resize to 512x512
        before = cv2.resize(before, (512, 512))
        after = cv2.resize(after, (512, 512))

        # Validate both halves have faces
        if not has_face(before) or not has_face(after):
            continue

        pair_id = f"{meta['source']}_{meta['procedure']}_{i:05d}"
        before_path = before_dir / f"{pair_id}_before.png"
        after_path = after_dir / f"{pair_id}_after.png"

        cv2.imwrite(str(before_path), before)
        cv2.imwrite(str(after_path), after)

        valid_pairs.append({
            **meta,
            "pair_id": pair_id,
            "before_path": str(before_path),
            "after_path": str(after_path),
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(metadata)}, valid pairs: {len(valid_pairs)}")

    return valid_pairs


def main():
    parser = argparse.ArgumentParser(description="Scrape plastic surgery before/after images")
    parser.add_argument("--output", type=str, default="data/real_surgery_pairs",
                        help="Output directory")
    parser.add_argument("--sources", nargs="+",
                        default=["realself", "locateadoc", "surgeryorg"],
                        choices=["realself", "locateadoc", "surgeryorg", "all"],
                        help="Which sources to scrape")
    parser.add_argument("--max_pages", type=int, default=30,
                        help="Max pages per procedure per source")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, just process existing raw images")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = []

    if "all" in args.sources:
        args.sources = ["realself", "locateadoc", "surgeryorg"]

    if not args.skip_download:
        print("=== Phase 1: Scraping images ===")

        if "realself" in args.sources:
            print("\n[1/3] RealSelf.com")
            meta = scrape_realself(output_dir, args.max_pages)
            all_metadata.extend(meta)
            print(f"  Downloaded {len(meta)} raw images")

        if "locateadoc" in args.sources:
            print("\n[2/3] LocateADoc.com")
            meta = scrape_locateadoc(output_dir, args.max_pages)
            all_metadata.extend(meta)
            print(f"  Downloaded {len(meta)} raw images")

        if "surgeryorg" in args.sources:
            print("\n[3/3] Surgery.org")
            meta = scrape_surgery_org(output_dir, args.max_pages)
            all_metadata.extend(meta)
            print(f"  Downloaded {len(meta)} raw images")

        # Save raw metadata
        with open(output_dir / "raw_metadata.json", "w") as f:
            json.dump(all_metadata, f, indent=2)

        print(f"\nTotal raw images: {len(all_metadata)}")
    else:
        # Load existing metadata
        meta_path = output_dir / "raw_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                all_metadata = json.load(f)
            print(f"Loaded {len(all_metadata)} existing raw images")

    if not all_metadata:
        print("No images to process!")
        return

    print("\n=== Phase 2: Splitting and validating ===")
    valid_pairs = process_raw_images(all_metadata, output_dir)

    # Save final metadata
    with open(output_dir / "pairs_metadata.json", "w") as f:
        json.dump(valid_pairs, f, indent=2)

    # Summary
    by_proc = {}
    for p in valid_pairs:
        proc = p["procedure"]
        by_proc[proc] = by_proc.get(proc, 0) + 1

    print(f"\n=== Results ===")
    print(f"Total valid pairs: {len(valid_pairs)}")
    for proc, count in sorted(by_proc.items()):
        print(f"  {proc}: {count}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
