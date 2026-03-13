"""Mass-scale before/after surgery image scraper.

Uses multiple search engines with hundreds of query variations per procedure
to collect as many unique before/after surgery images as possible.

Target: maximize unique images per procedure type.
Deduplication via perceptual hashing to avoid duplicates across queries.
"""

import argparse
import hashlib
import json
import os
import struct
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# Try imports
try:
    from icrawler.builtin import BingImageCrawler, GoogleImageCrawler, BaiduImageCrawler
    HAS_ICRAWLER = True
except ImportError:
    HAS_ICRAWLER = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


def phash(img: np.ndarray, hash_size: int = 8) -> str:
    """Perceptual hash for deduplication."""
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    return ''.join(['1' if b else '0' for row in diff for b in row])


def hamming_distance(h1: str, h2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


# ============================================================
# QUERY GENERATION — hundreds of variations per procedure
# ============================================================

def generate_queries(procedure: str) -> list[str]:
    """Generate 100+ search query variations for a procedure."""
    queries = []

    # Base terms by procedure
    if procedure == "rhinoplasty":
        primary_terms = [
            "rhinoplasty", "nose job", "nose surgery", "nose reshaping",
            "nasal surgery", "septorhinoplasty", "nose reduction",
            "nose augmentation", "revision rhinoplasty", "ethnic rhinoplasty",
            "asian rhinoplasty", "african american rhinoplasty",
            "hispanic rhinoplasty", "bulbous nose surgery",
            "dorsal hump removal", "nasal tip refinement", "alar reduction",
            "nostril reduction", "bridge augmentation", "primary rhinoplasty",
            "open rhinoplasty", "closed rhinoplasty", "tip plasty",
            "nasal reconstruction", "cosmetic nose surgery",
        ]
    elif procedure == "blepharoplasty":
        primary_terms = [
            "blepharoplasty", "eyelid surgery", "eyelid lift", "eye lift",
            "upper blepharoplasty", "lower blepharoplasty",
            "upper eyelid surgery", "lower eyelid surgery",
            "eye bag removal", "under eye surgery", "droopy eyelid surgery",
            "asian eyelid surgery", "double eyelid surgery",
            "ptosis repair", "eyelid rejuvenation", "periorbital surgery",
            "transconjunctival blepharoplasty", "cosmetic eyelid surgery",
            "hooded eyelid surgery", "eye bag treatment surgery",
            "eyelid tuck", "brow lift eyelid",
        ]
    elif procedure == "rhytidectomy":
        primary_terms = [
            "facelift", "face lift", "rhytidectomy", "mini facelift",
            "deep plane facelift", "SMAS facelift", "neck lift",
            "lower facelift", "mid facelift", "full facelift",
            "thread lift face", "jowl lift", "jawline surgery facelift",
            "facial rejuvenation surgery", "face tightening surgery",
            "ponytail facelift", "short scar facelift",
            "endoscopic facelift", "liquid facelift surgery",
            "vertical facelift", "composite facelift",
        ]
    elif procedure == "orthognathic":
        primary_terms = [
            "orthognathic surgery", "jaw surgery", "corrective jaw surgery",
            "mandibular surgery", "maxillary surgery", "jaw advancement",
            "jaw reduction", "chin surgery", "genioplasty",
            "jaw reconstruction", "underbite surgery", "overbite surgery",
            "jaw realignment", "double jaw surgery", "bimaxillary surgery",
            "le fort osteotomy", "BSSO surgery", "mandibular osteotomy",
            "maxillary osteotomy", "jaw contouring surgery",
            "v line jaw surgery", "jaw angle surgery",
            "chin augmentation surgery", "chin reduction surgery",
        ]
    else:
        primary_terms = [procedure]

    # Suffix variations
    suffixes = [
        "before and after",
        "before after results",
        "before after photos",
        "results comparison",
        "patient results",
        "results gallery",
        "before after front view",
        "before after side view",
        "real results",
        "transformation photos",
        "clinical results",
        "plastic surgery results",
        "cosmetic surgery results",
        "surgery outcome photos",
        "pre post operative",
        "preoperative postoperative",
    ]

    # Demographic/specificity modifiers
    modifiers = [
        "", "male", "female", "young", "40s", "50s", "60s",
        "natural results", "dramatic results", "subtle results",
        "best results", "celebrity", "real patient",
    ]

    # Generate combinations
    for term in primary_terms:
        for suffix in suffixes:
            queries.append(f"{term} {suffix}")

    # Add modifier combinations (subset to avoid too many)
    for term in primary_terms[:10]:
        for mod in modifiers[1:]:  # skip empty
            queries.append(f"{mod} {term} before after")

    # Doctor/clinic gallery queries
    doctor_queries = [
        f"{procedure} surgeon gallery",
        f"{procedure} doctor results",
        f"{procedure} clinic before after",
        f"board certified {procedure} results",
        f"{procedure} specialist before after photos",
    ]
    queries.extend(doctor_queries)

    # Location-based queries for diversity
    locations = [
        "new york", "los angeles", "miami", "beverly hills",
        "london", "istanbul", "seoul", "bangkok", "dubai",
        "chicago", "houston", "san francisco", "toronto",
    ]
    for loc in locations:
        queries.append(f"{primary_terms[0]} before after {loc}")

    # Deduplicate and shuffle
    seen = set()
    unique = []
    for q in queries:
        q_lower = q.lower().strip()
        if q_lower not in seen:
            seen.add(q_lower)
            unique.append(q)

    return unique


# ============================================================
# DIRECT GALLERY SCRAPING
# ============================================================

GALLERY_URLS = {
    "rhinoplasty": [
        # Clinic gallery pages that serve images without JS
        "https://www.drgal.com/photo-gallery/rhinoplasty",
        "https://www.realself.com/rhinoplasty/before-and-after-photos",
        "https://www.zwivel.com/before-after/rhinoplasty",
    ],
    "blepharoplasty": [
        "https://www.realself.com/eyelid-surgery/before-and-after-photos",
        "https://www.zwivel.com/before-after/blepharoplasty",
    ],
    "rhytidectomy": [
        "https://www.realself.com/facelift/before-and-after-photos",
        "https://www.zwivel.com/before-after/facelift",
    ],
    "orthognathic": [
        "https://www.realself.com/jaw-surgery/before-and-after-photos",
    ],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/*,*/*;q=0.8",
}


def scrape_gallery_page(url: str, output_dir: Path, proc: str, seen_hashes: set) -> int:
    """Scrape images from a gallery page."""
    if not HAS_BS4:
        return 0

    count = 0
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return 0

    soup = BeautifulSoup(html, "html.parser")

    # Find all image tags
    img_tags = soup.find_all("img")
    for img_tag in img_tags:
        src = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-lazy-src")
        if not src:
            continue
        if not src.startswith("http"):
            continue
        # Filter for likely before/after images (skip icons, logos, etc.)
        if any(skip in src.lower() for skip in ["icon", "logo", "avatar", "sprite", "pixel"]):
            continue

        try:
            img_hash = hashlib.md5(src.encode()).hexdigest()[:16]
            save_path = output_dir / f"gallery_{proc}_{img_hash}.jpg"
            if save_path.exists():
                continue

            req = urllib.request.Request(src, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            save_path.write_bytes(data)

            # Check if it's a valid image and compute phash
            test_img = cv2.imread(str(save_path))
            if test_img is None or test_img.shape[0] < 100 or test_img.shape[1] < 100:
                save_path.unlink(missing_ok=True)
                continue

            h = phash(test_img)
            if h in seen_hashes:
                save_path.unlink(missing_ok=True)
                continue
            seen_hashes.add(h)
            count += 1

        except Exception:
            continue

    return count


# ============================================================
# MAIN CRAWLING
# ============================================================

def crawl_procedure(proc: str, output_dir: Path, target: int = 50000) -> int:
    """Crawl as many images as possible for a procedure."""
    proc_dir = output_dir / proc
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Load existing hashes for dedup
    seen_hashes = set()
    existing_files = list(proc_dir.glob("*"))
    print(f"  Loading {len(existing_files)} existing images for dedup...")
    for f in existing_files:
        try:
            img = cv2.imread(str(f))
            if img is not None:
                seen_hashes.add(phash(img))
        except Exception:
            pass

    initial_count = len(existing_files)
    print(f"  {proc}: starting with {initial_count} images, target {target}")

    queries = generate_queries(proc)
    print(f"  Generated {len(queries)} search queries")

    # Phase 1: Bing Image Search (primary source)
    print(f"\n  --- Phase 1: Bing Image Search ---")
    bing_count = 0
    for i, query in enumerate(queries):
        current = len(list(proc_dir.glob("*")))
        if current >= target:
            print(f"  Target reached! ({current}/{target})")
            break

        if (i + 1) % 10 == 0:
            current = len(list(proc_dir.glob("*")))
            print(f"  [{i+1}/{len(queries)}] Bing queries done, {current} total images")

        try:
            crawler = BingImageCrawler(
                storage={'root_dir': str(proc_dir)},
                log_level=50,  # CRITICAL only
            )
            # Request more per query since many will be duplicates
            crawler.crawl(
                keyword=query,
                max_num=200,
                min_size=(200, 200),
                file_idx_offset='auto',
            )
        except Exception as e:
            pass

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    current_count = len(list(proc_dir.glob("*")))
    print(f"  Bing phase complete: {initial_count} -> {current_count} (+{current_count - initial_count})")

    # Phase 2: Google Image Search
    print(f"\n  --- Phase 2: Google Image Search ---")
    for i, query in enumerate(queries[:50]):  # Google is stricter, use fewer queries
        current = len(list(proc_dir.glob("*")))
        if current >= target:
            break

        if (i + 1) % 10 == 0:
            current = len(list(proc_dir.glob("*")))
            print(f"  [{i+1}/50] Google queries done, {current} total images")

        try:
            crawler = GoogleImageCrawler(
                storage={'root_dir': str(proc_dir)},
                log_level=50,
            )
            crawler.crawl(
                keyword=query,
                max_num=100,
                min_size=(200, 200),
                file_idx_offset='auto',
            )
        except Exception:
            pass

        time.sleep(0.5)

    current_count = len(list(proc_dir.glob("*")))
    print(f"  Google phase complete: {current_count} total images")

    # Phase 3: Direct gallery scraping
    print(f"\n  --- Phase 3: Gallery scraping ---")
    gallery_urls = GALLERY_URLS.get(proc, [])
    for url in gallery_urls:
        n = scrape_gallery_page(url, proc_dir, proc, seen_hashes)
        if n > 0:
            print(f"    {url}: +{n} images")

    # Phase 4: Baidu Image Search (Chinese medical photos)
    print(f"\n  --- Phase 4: Baidu Image Search ---")
    baidu_terms = {
        "rhinoplasty": ["鼻整形手术前后对比", "隆鼻手术效果", "鼻部整形前后"],
        "blepharoplasty": ["双眼皮手术前后对比", "眼部整形效果", "开眼角前后"],
        "rhytidectomy": ["拉皮手术前后对比", "面部提升术效果", "除皱手术前后"],
        "orthognathic": ["正颌手术前后对比", "下颌骨手术效果", "颌面手术前后"],
    }
    for query in baidu_terms.get(proc, []):
        try:
            crawler = BaiduImageCrawler(
                storage={'root_dir': str(proc_dir)},
                log_level=50,
            )
            crawler.crawl(
                keyword=query,
                max_num=200,
                file_idx_offset='auto',
            )
        except Exception:
            pass
        time.sleep(0.5)

    final_count = len(list(proc_dir.glob("*")))
    print(f"\n  {proc} FINAL: {final_count} images (was {initial_count})")
    return final_count


def main():
    parser = argparse.ArgumentParser(description="Mass scrape surgery before/after images")
    parser.add_argument("--output", type=str, default="data/real_surgery_pairs/raw",
                        help="Output directory for raw images")
    parser.add_argument("--target", type=int, default=50000,
                        help="Target images per procedure")
    parser.add_argument("--procedures", nargs="+",
                        default=["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic"])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Mass Surgery Image Scraper")
    print(f"Target: {args.target} images per procedure")
    print(f"Procedures: {args.procedures}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    results = {}
    for proc in args.procedures:
        print(f"\n{'='*60}")
        print(f"PROCEDURE: {proc}")
        print(f"{'='*60}")
        count = crawl_procedure(proc, output_dir, args.target)
        results[proc] = count

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    total = 0
    for proc, count in results.items():
        print(f"  {proc}: {count} images")
        total += count
    print(f"  TOTAL: {total} images")

    # Save metadata
    with open(output_dir / "scrape_metadata.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
