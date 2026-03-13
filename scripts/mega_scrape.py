"""Mega-scale before/after surgery image scraper.

Uses Selenium headless Chrome + direct HTTP + search engines to scrape
from every available source: RealSelf, Reddit, Pinterest, clinic sites,
Bing, Google, Baidu, and more.

Target: 50K images per procedure.
"""

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# icrawler
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler, BaiduImageCrawler

from bs4 import BeautifulSoup

# ============================================================
# CONFIG
# ============================================================

WORK_DIR = Path(__file__).resolve().parent.parent
CHROME_BIN = str(WORK_DIR / "tools" / "chrome" / "chrome-bin" / "chrome")
CHROMEDRIVER = str(WORK_DIR / "tools" / "chrome" / "chromedriver")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

PROCEDURES = {
    "rhinoplasty": {
        "realself_slug": "rhinoplasty",
        "reddit_subs": ["PlasticSurgery", "Rhinoplasty", "NoseJob", "cosmeticsurgery",
                        "plasticsurgery", "beforeandafter"],
        "reddit_terms": ["rhinoplasty", "nose job", "nose surgery"],
        "search_terms": [
            "rhinoplasty", "nose job", "nose surgery", "nose reshaping",
            "nasal surgery", "septorhinoplasty", "nose reduction",
            "revision rhinoplasty", "ethnic rhinoplasty", "tip plasty",
            "alar reduction", "dorsal hump removal", "open rhinoplasty",
            "closed rhinoplasty", "nose augmentation", "bulbous nose surgery",
            "asian rhinoplasty", "african american rhinoplasty",
            "primary rhinoplasty", "cosmetic nose surgery",
        ],
    },
    "blepharoplasty": {
        "realself_slug": "eyelid-surgery",
        "reddit_subs": ["PlasticSurgery", "cosmeticsurgery", "plasticsurgery",
                        "beforeandafter"],
        "reddit_terms": ["blepharoplasty", "eyelid surgery", "eye lift", "eye bag"],
        "search_terms": [
            "blepharoplasty", "eyelid surgery", "eyelid lift", "eye lift",
            "upper blepharoplasty", "lower blepharoplasty", "eye bag removal",
            "droopy eyelid surgery", "double eyelid surgery", "ptosis repair",
            "hooded eyelid surgery", "asian eyelid surgery",
            "transconjunctival blepharoplasty", "eyelid rejuvenation",
            "cosmetic eyelid surgery", "eyelid tuck",
        ],
    },
    "rhytidectomy": {
        "realself_slug": "facelift",
        "reddit_subs": ["PlasticSurgery", "cosmeticsurgery", "plasticsurgery",
                        "beforeandafter", "30PlusSkinCare"],
        "reddit_terms": ["facelift", "face lift", "rhytidectomy", "neck lift"],
        "search_terms": [
            "facelift", "face lift", "rhytidectomy", "mini facelift",
            "deep plane facelift", "SMAS facelift", "neck lift",
            "lower facelift", "mid facelift", "full facelift",
            "jowl lift", "facial rejuvenation surgery", "ponytail facelift",
            "short scar facelift", "endoscopic facelift", "thread lift",
            "vertical facelift", "composite facelift",
        ],
    },
    "orthognathic": {
        "realself_slug": "jaw-surgery",
        "reddit_subs": ["jawsurgery", "orthognathics", "PlasticSurgery",
                        "cosmeticsurgery", "beforeandafter"],
        "reddit_terms": ["jaw surgery", "orthognathic", "genioplasty", "underbite",
                         "overbite"],
        "search_terms": [
            "orthognathic surgery", "jaw surgery", "corrective jaw surgery",
            "mandibular surgery", "jaw advancement", "jaw reduction",
            "genioplasty", "underbite surgery", "overbite surgery",
            "double jaw surgery", "bimaxillary surgery", "chin surgery",
            "v line jaw surgery", "jaw contouring", "BSSO surgery",
            "le fort osteotomy", "jaw realignment",
        ],
    },
}

# ============================================================
# SELENIUM DRIVER
# ============================================================

def make_driver() -> webdriver.Chrome:
    """Create a headless Chrome driver."""
    options = Options()
    options.binary_location = CHROME_BIN
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    service = Service(CHROMEDRIVER)
    return webdriver.Chrome(service=service, options=options)


def download_image(url: str, save_path: Path, timeout: int = 15) -> bool:
    """Download image from URL."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if len(data) < 1000:  # too small to be a real image
                return False
        save_path.write_bytes(data)
        # Validate it's actually an image
        img = cv2.imread(str(save_path))
        if img is None or img.shape[0] < 80 or img.shape[1] < 80:
            save_path.unlink(missing_ok=True)
            return False
        return True
    except Exception:
        save_path.unlink(missing_ok=True)
        return False


def img_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


# ============================================================
# SOURCE 1: REALSELF (Selenium — JS rendered)
# ============================================================

def scrape_realself(proc: str, slug: str, output_dir: Path, max_pages: int = 200) -> int:
    """Scrape RealSelf before/after photos using Selenium."""
    print(f"  [RealSelf] Scraping {proc} ({slug})...")
    count = 0
    driver = None

    try:
        driver = make_driver()

        for page in range(1, max_pages + 1):
            url = f"https://www.realself.com/photos/{slug}?page={page}"
            try:
                driver.get(url)
                time.sleep(2)

                # Scroll to load lazy images
                for _ in range(5):
                    driver.execute_script("window.scrollBy(0, 1000)")
                    time.sleep(0.5)

                # Find all image elements
                img_elements = driver.find_elements(By.TAG_NAME, "img")
                img_urls = []
                for el in img_elements:
                    src = el.get_attribute("src") or el.get_attribute("data-src")
                    if src and ("realself" in src or "cloudfront" in src):
                        if any(ext in src.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                            img_urls.append(src)

                # Also check for background images in divs
                photo_divs = driver.find_elements(By.CSS_SELECTOR, "[style*='background-image']")
                for div in photo_divs:
                    style = div.get_attribute("style")
                    urls = re.findall(r"url\(['\"]?(https?://[^'\")\s]+)['\"]?\)", style)
                    img_urls.extend(urls)

                if not img_urls:
                    # Try looking for srcset
                    img_elements = driver.find_elements(By.CSS_SELECTOR, "img[srcset]")
                    for el in img_elements:
                        srcset = el.get_attribute("srcset")
                        if srcset:
                            urls = re.findall(r"(https?://[^\s,]+)", srcset)
                            img_urls.extend(urls)

                img_urls = list(set(img_urls))

                if not img_urls:
                    if page > 3:  # Give first few pages a chance
                        break
                    continue

                for img_url in img_urls:
                    h = img_hash(img_url)
                    save_path = output_dir / f"realself_{proc}_{h}.jpg"
                    if save_path.exists():
                        continue
                    if download_image(img_url, save_path):
                        count += 1

                if page % 10 == 0:
                    print(f"    Page {page}: {count} images so far")

            except Exception as e:
                if page > 5:
                    break
                continue

            time.sleep(1.0)

    except Exception as e:
        print(f"    RealSelf error: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    print(f"    RealSelf {proc}: {count} images")
    return count


# ============================================================
# SOURCE 2: REDDIT (via old.reddit.com — works without JS)
# ============================================================

def scrape_reddit(proc: str, config: dict, output_dir: Path) -> int:
    """Scrape Reddit before/after posts."""
    print(f"  [Reddit] Scraping {proc}...")
    count = 0
    seen_urls = set()

    subs = config["reddit_subs"]
    terms = config["reddit_terms"]

    for sub in subs:
        for term in terms:
            # Search via old.reddit which works without JS
            for sort in ["relevance", "top", "new"]:
                for t_range in ["all", "year", "month"]:
                    url = (f"https://old.reddit.com/r/{sub}/search?"
                           f"q={term.replace(' ', '+')}+before+after&"
                           f"restrict_sr=on&sort={sort}&t={t_range}")

                    try:
                        req = urllib.request.Request(url, headers=HEADERS)
                        with urllib.request.urlopen(req, timeout=15) as resp:
                            html = resp.read().decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                    # Find all image URLs in the page
                    patterns = [
                        r'(https?://i\.redd\.it/[^\s"\'<>]+\.(?:jpg|jpeg|png|webp))',
                        r'(https?://preview\.redd\.it/[^\s"\'<>]+\.(?:jpg|jpeg|png|webp))',
                        r'(https?://i\.imgur\.com/[^\s"\'<>]+\.(?:jpg|jpeg|png|webp))',
                        r'(https?://external-preview\.redd\.it/[^\s"\'<>]+)',
                    ]

                    img_urls = []
                    for pat in patterns:
                        img_urls.extend(re.findall(pat, html, re.IGNORECASE))

                    img_urls = [u for u in set(img_urls) if u not in seen_urls]
                    seen_urls.update(img_urls)

                    for img_url in img_urls:
                        # Clean up reddit preview URLs
                        img_url = img_url.split("?")[0]
                        if "preview.redd.it" in img_url:
                            img_url = img_url.replace("preview.redd.it", "i.redd.it")

                        h = img_hash(img_url)
                        save_path = output_dir / f"reddit_{proc}_{h}.jpg"
                        if save_path.exists():
                            continue
                        if download_image(img_url, save_path):
                            count += 1

                    time.sleep(1.5)

        # Also scrape the subreddit directly (top posts)
        for time_filter in ["all", "year", "month"]:
            url = f"https://old.reddit.com/r/{sub}/top/?t={time_filter}"
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
            except Exception:
                continue

            for pat in patterns:
                for img_url in re.findall(pat, html, re.IGNORECASE):
                    img_url = img_url.split("?")[0]
                    if img_url in seen_urls:
                        continue
                    seen_urls.add(img_url)
                    h = img_hash(img_url)
                    save_path = output_dir / f"reddit_{proc}_{h}.jpg"
                    if save_path.exists():
                        continue
                    if download_image(img_url, save_path):
                        count += 1

            time.sleep(1.5)

    print(f"    Reddit {proc}: {count} images")
    return count


# ============================================================
# SOURCE 3: REALSELF REVIEWS (direct HTTP scraping of review pages)
# ============================================================

def scrape_realself_reviews(proc: str, slug: str, output_dir: Path, max_reviews: int = 500) -> int:
    """Scrape individual RealSelf review pages for images."""
    print(f"  [RealSelf Reviews] Scraping {proc}...")
    count = 0
    driver = None

    try:
        driver = make_driver()

        # Navigate to reviews listing
        for page in range(1, 50):
            url = f"https://www.realself.com/{slug}/reviews?page={page}"
            try:
                driver.get(url)
                time.sleep(2)

                # Find review links
                links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/review/']")
                review_urls = list(set(l.get_attribute("href") for l in links if l.get_attribute("href")))

                if not review_urls:
                    break

                for review_url in review_urls[:20]:  # limit per page
                    try:
                        driver.get(review_url)
                        time.sleep(1.5)

                        # Scroll to load images
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2)")
                        time.sleep(1)

                        img_elements = driver.find_elements(By.TAG_NAME, "img")
                        for el in img_elements:
                            src = el.get_attribute("src") or el.get_attribute("data-src")
                            if not src:
                                continue
                            if any(skip in src for skip in ["avatar", "icon", "logo", "sprite"]):
                                continue
                            if not any(ext in src.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                                continue

                            h = img_hash(src)
                            save_path = output_dir / f"rsreview_{proc}_{h}.jpg"
                            if save_path.exists():
                                continue
                            if download_image(src, save_path):
                                count += 1

                    except Exception:
                        continue

                if page % 5 == 0:
                    print(f"    Review page {page}: {count} images")

            except Exception:
                continue

    except Exception as e:
        print(f"    RealSelf Reviews error: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    print(f"    RealSelf Reviews {proc}: {count} images")
    return count


# ============================================================
# SOURCE 4: PINTEREST (Selenium)
# ============================================================

def scrape_pinterest(proc: str, search_terms: list[str], output_dir: Path) -> int:
    """Scrape Pinterest for surgery before/after images."""
    print(f"  [Pinterest] Scraping {proc}...")
    count = 0
    driver = None

    try:
        driver = make_driver()

        for term in search_terms[:8]:
            query = f"{term} before after results"
            url = f"https://www.pinterest.com/search/pins/?q={query.replace(' ', '%20')}"

            try:
                driver.get(url)
                time.sleep(3)

                # Scroll to load more
                for _ in range(15):
                    driver.execute_script("window.scrollBy(0, 2000)")
                    time.sleep(1)

                img_elements = driver.find_elements(By.TAG_NAME, "img")
                for el in img_elements:
                    src = el.get_attribute("src") or el.get_attribute("data-src")
                    if not src:
                        continue
                    # Pinterest serves images via pinimg.com
                    if "pinimg.com" not in src and "pinterest" not in src:
                        continue
                    # Get higher res version
                    src = re.sub(r"/\d+x/", "/736x/", src)

                    h = img_hash(src)
                    save_path = output_dir / f"pinterest_{proc}_{h}.jpg"
                    if save_path.exists():
                        continue
                    if download_image(src, save_path):
                        count += 1

            except Exception:
                continue

            time.sleep(2)

    except Exception as e:
        print(f"    Pinterest error: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    print(f"    Pinterest {proc}: {count} images")
    return count


# ============================================================
# SOURCE 5: CLINIC GALLERY SITES (Selenium for JS-rendered)
# ============================================================

CLINIC_GALLERIES = {
    "rhinoplasty": [
        "https://www.drrundlewalker.com.au/before-after/rhinoplasty/",
        "https://www.beverlyhillsplasticsurgery.com/rhinoplasty-before-and-after/",
        "https://www.drghavami.com/rhinoplasty-before-after-photos/",
        "https://www.drdonyoo.com/rhinoplasty-before-and-after-photos/",
        "https://www.zwivel.com/before-after/rhinoplasty",
    ],
    "blepharoplasty": [
        "https://www.zwivel.com/before-after/blepharoplasty",
        "https://www.beverlyhillsplasticsurgery.com/eyelid-surgery-before-and-after/",
    ],
    "rhytidectomy": [
        "https://www.zwivel.com/before-after/facelift",
        "https://www.beverlyhillsplasticsurgery.com/facelift-before-and-after/",
    ],
    "orthognathic": [
        "https://www.zwivel.com/before-after/jaw-surgery",
    ],
}


def scrape_clinic_galleries(proc: str, output_dir: Path) -> int:
    """Scrape clinic gallery sites."""
    print(f"  [Clinic Galleries] Scraping {proc}...")
    count = 0
    galleries = CLINIC_GALLERIES.get(proc, [])

    driver = None
    try:
        driver = make_driver()

        for gallery_url in galleries:
            try:
                driver.get(gallery_url)
                time.sleep(3)

                # Scroll through gallery
                for _ in range(10):
                    driver.execute_script("window.scrollBy(0, 1500)")
                    time.sleep(0.5)

                img_elements = driver.find_elements(By.TAG_NAME, "img")
                for el in img_elements:
                    src = el.get_attribute("src") or el.get_attribute("data-src") or el.get_attribute("data-lazy-src")
                    if not src or not src.startswith("http"):
                        continue
                    if any(skip in src.lower() for skip in ["icon", "logo", "avatar", "sprite", "pixel", "1x1"]):
                        continue

                    h = img_hash(src)
                    save_path = output_dir / f"clinic_{proc}_{h}.jpg"
                    if save_path.exists():
                        continue
                    if download_image(src, save_path):
                        count += 1

            except Exception:
                continue

    except Exception as e:
        print(f"    Clinic error: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    print(f"    Clinic Galleries {proc}: {count} images")
    return count


# ============================================================
# SOURCE 6: SEARCH ENGINES (icrawler — Bing, Google, Baidu)
# ============================================================

def search_engine_crawl(proc: str, search_terms: list[str], output_dir: Path, target: int = 50000) -> int:
    """Massive search engine image crawling."""
    print(f"  [Search Engines] Scraping {proc}...")
    initial = len(list(output_dir.glob("*")))

    suffixes = [
        "before and after", "before after results", "before after photos",
        "results comparison", "patient results", "results gallery",
        "before after front view", "before after side view",
        "real results", "transformation photos", "clinical results",
        "pre post operative", "preoperative postoperative",
        "surgery outcome photos", "plastic surgery results",
        "cosmetic surgery results", "before after 2024",
        "before after 2023", "before after 2025",
    ]

    modifiers = [
        "", "male", "female", "natural results", "dramatic results",
        "best results", "real patient", "board certified",
    ]

    locations = [
        "new york", "los angeles", "miami", "beverly hills",
        "london", "istanbul", "seoul", "bangkok", "dubai",
        "chicago", "houston", "san francisco", "toronto",
        "sydney", "melbourne", "mumbai", "delhi",
    ]

    # Build query list
    queries = []
    for term in search_terms:
        for suffix in suffixes:
            queries.append(f"{term} {suffix}")
        for mod in modifiers[1:]:
            queries.append(f"{mod} {term} before after")
        for loc in locations:
            queries.append(f"{term} before after {loc}")

    # Deduplicate
    seen = set()
    unique_queries = []
    for q in queries:
        ql = q.lower().strip()
        if ql not in seen:
            seen.add(ql)
            unique_queries.append(q)

    print(f"    {len(unique_queries)} unique search queries")

    # Bing (primary)
    for i, query in enumerate(unique_queries):
        current = len(list(output_dir.glob("*")))
        if current >= target:
            break

        if (i + 1) % 25 == 0:
            current = len(list(output_dir.glob("*")))
            print(f"    Bing [{i+1}/{len(unique_queries)}] {current} total images")

        try:
            crawler = BingImageCrawler(
                storage={'root_dir': str(output_dir)},
                log_level=50,
            )
            crawler.crawl(keyword=query, max_num=200, min_size=(200, 200),
                          file_idx_offset='auto')
        except Exception:
            pass
        time.sleep(0.2)

    # Google (secondary — stricter rate limits)
    for i, query in enumerate(unique_queries[:100]):
        current = len(list(output_dir.glob("*")))
        if current >= target:
            break

        if (i + 1) % 20 == 0:
            current = len(list(output_dir.glob("*")))
            print(f"    Google [{i+1}/100] {current} total images")

        try:
            crawler = GoogleImageCrawler(
                storage={'root_dir': str(output_dir)},
                log_level=50,
            )
            crawler.crawl(keyword=query, max_num=100, min_size=(200, 200),
                          file_idx_offset='auto')
        except Exception:
            pass
        time.sleep(0.5)

    # Baidu (Chinese medical photos)
    baidu_terms = {
        "rhinoplasty": ["鼻整形手术前后对比", "隆鼻手术效果照片", "鼻部整形前后", "鼻子整形前后",
                        "隆鼻前后对比图", "鼻整形案例", "鼻综合手术前后"],
        "blepharoplasty": ["双眼皮手术前后对比", "眼部整形效果", "开眼角前后", "眼袋去除前后",
                           "割双眼皮前后对比", "眼部整形案例"],
        "rhytidectomy": ["拉皮手术前后对比", "面部提升术效果", "除皱手术前后", "拉皮手术案例",
                         "面部抗衰手术前后"],
        "orthognathic": ["正颌手术前后对比", "下颌骨手术效果", "颌面手术前后", "下巴整形前后",
                         "正颌正畸前后对比"],
    }
    for query in baidu_terms.get(proc, []):
        try:
            crawler = BaiduImageCrawler(
                storage={'root_dir': str(output_dir)},
                log_level=50,
            )
            crawler.crawl(keyword=query, max_num=300, file_idx_offset='auto')
        except Exception:
            pass
        time.sleep(0.3)

    final = len(list(output_dir.glob("*")))
    print(f"    Search Engines {proc}: {initial} -> {final} (+{final - initial})")
    return final - initial


# ============================================================
# SOURCE 7: IMGUR ALBUMS (direct HTTP)
# ============================================================

def scrape_imgur(proc: str, output_dir: Path) -> int:
    """Search Imgur for surgery before/after albums."""
    print(f"  [Imgur] Scraping {proc}...")
    count = 0

    search_terms = PROCEDURES[proc]["search_terms"][:5]

    for term in search_terms:
        query = f"{term} before after"
        url = f"https://imgur.com/search?q={query.replace(' ', '+')}"

        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # Find imgur image URLs
            img_urls = re.findall(
                r'(https?://i\.imgur\.com/[a-zA-Z0-9]+\.(?:jpg|jpeg|png|webp))',
                html, re.IGNORECASE
            )
            img_urls = list(set(img_urls))

            for img_url in img_urls:
                h = img_hash(img_url)
                save_path = output_dir / f"imgur_{proc}_{h}.jpg"
                if save_path.exists():
                    continue
                if download_image(img_url, save_path):
                    count += 1

        except Exception:
            continue

        time.sleep(1)

    print(f"    Imgur {proc}: {count} images")
    return count


# ============================================================
# MAIN
# ============================================================

def scrape_procedure(proc: str, output_dir: Path, target: int = 50000) -> dict:
    """Run all scraping sources for a single procedure."""
    proc_dir = output_dir / proc
    proc_dir.mkdir(parents=True, exist_ok=True)

    config = PROCEDURES[proc]
    initial = len(list(proc_dir.glob("*")))
    results = {"procedure": proc, "initial": initial}

    print(f"\n{'='*60}")
    print(f"PROCEDURE: {proc} (existing: {initial}, target: {target})")
    print(f"{'='*60}")

    # Run all sources
    results["reddit"] = scrape_reddit(proc, config, proc_dir)
    results["realself"] = scrape_realself(proc, config["realself_slug"], proc_dir)
    results["realself_reviews"] = scrape_realself_reviews(proc, config["realself_slug"], proc_dir)
    results["pinterest"] = scrape_pinterest(proc, config["search_terms"], proc_dir)
    results["clinic"] = scrape_clinic_galleries(proc, proc_dir)
    results["imgur"] = scrape_imgur(proc, proc_dir)

    # Check if we need more from search engines
    current = len(list(proc_dir.glob("*")))
    remaining = target - current
    if remaining > 0:
        results["search_engines"] = search_engine_crawl(
            proc, config["search_terms"], proc_dir, target
        )

    final = len(list(proc_dir.glob("*")))
    results["final"] = final
    results["gained"] = final - initial

    print(f"\n  {proc} COMPLETE: {final} images (gained {final - initial})")
    return results


def main():
    parser = argparse.ArgumentParser(description="Mega surgery image scraper")
    parser.add_argument("--output", type=str, default="data/real_surgery_pairs/raw")
    parser.add_argument("--target", type=int, default=50000)
    parser.add_argument("--procedures", nargs="+",
                        default=["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic"])
    parser.add_argument("--procedure", type=str, default=None,
                        help="Single procedure to scrape (for parallel SLURM jobs)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.procedure:
        procedures = [args.procedure]
    else:
        procedures = args.procedures

    print("=" * 60)
    print(f"MEGA SURGERY IMAGE SCRAPER")
    print(f"Target: {args.target} images per procedure")
    print(f"Procedures: {procedures}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    all_results = {}
    for proc in procedures:
        if proc not in PROCEDURES:
            print(f"Unknown procedure: {proc}")
            continue
        result = scrape_procedure(proc, output_dir, args.target)
        all_results[proc] = result

    # Save results
    with open(output_dir / "mega_scrape_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    total = 0
    for proc, result in all_results.items():
        print(f"  {proc}: {result['final']} images (gained {result['gained']})")
        total += result["final"]
    print(f"  TOTAL: {total}")


if __name__ == "__main__":
    main()
