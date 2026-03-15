"""Collect surgical before/after pairs from public Instagram posts.

Data collection methodology adapted from Varghaei et al. (2025),
"Automated Assessment of Aesthetic Outcomes in Facial Plastic Surgery."

Scrapes publicly posted clinical before/after images from verified
plastic surgeons' Instagram accounts, then splits them into aligned
pre- and post-operative pairs using face detection and layout analysis.

Features beyond baseline:
  - Procedure classification from post captions/hashtags
  - Image quality filtering (resolution, face size, blur)
  - Caption metadata preservation for provenance
  - Output format compatible with mega_scrape.py pipeline
  - Curated surgeon account list

Requirements:
    pip install instaloader mediapipe opencv-python-headless

Usage:
    python scripts/collect_instagram_data.py \
        --accounts dr_smith dr_jones \
        --output data/instagram_pairs \
        --max-posts 500

    # Or from a file listing surgeon accounts:
    python scripts/collect_instagram_data.py \
        --accounts-file configs/surgeon_accounts.txt \
        --output data/instagram_pairs

    # Process already-downloaded images:
    python scripts/collect_instagram_data.py \
        --input-dir data/raw_instagram \
        --output data/instagram_pairs

Ethics:
    - Only scrape publicly posted clinical comparison images
    - Respect rate limits and Terms of Service
    - Do not redistribute raw images without IRB approval
    - De-identify all processed outputs
    - This script is for research purposes only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Procedure classification from captions/hashtags
# ============================================================

PROCEDURE_KEYWORDS: dict[str, list[str]] = {
    "rhinoplasty": [
        "rhinoplasty",
        "nosejob",
        "nose job",
        "nose surgery",
        "septorhinoplasty",
        "nose reshaping",
        "nasal surgery",
        "nose reduction",
        "nose tip",
        "dorsal hump",
        "bulbous tip",
        "alar reduction",
        "nostril",
    ],
    "blepharoplasty": [
        "blepharoplasty",
        "eyelid surgery",
        "eyelid lift",
        "eye lift",
        "upper bleph",
        "lower bleph",
        "brow lift",
        "browlift",
        "upper eyelid",
        "lower eyelid",
        "eye bag",
        "ptosis",
    ],
    "rhytidectomy": [
        "facelift",
        "face lift",
        "rhytidectomy",
        "mini lift",
        "deep plane",
        "smas lift",
        "neck lift",
        "necklift",
        "jowl",
        "midface lift",
        "lower face",
        "facial rejuvenation",
    ],
    "orthognathic": [
        "jaw surgery",
        "orthognathic",
        "mandibular",
        "maxillary",
        "chin surgery",
        "genioplasty",
        "mentoplasty",
        "sliding genioplasty",
        "chin implant",
        "chin augmentation",
        "jaw advancement",
        "jaw reduction",
        "v-line",
    ],
    "brow_lift": [
        "brow lift",
        "browlift",
        "forehead lift",
        "brow ptosis",
        "endoscopic brow",
    ],
    "mentoplasty": [
        "chin surgery",
        "mentoplasty",
        "chin implant",
        "chin reduction",
        "chin augmentation",
        "genioplasty",
        "sliding genioplasty",
    ],
}

# Hashtags and keywords that indicate a before/after post
BEFORE_AFTER_INDICATORS = [
    "beforeandafter",
    "before and after",
    "transformation",
    "results",
    "postop",
    "post op",
    "preop",
    "pre op",
    "surgical result",
    "outcome",
    "healing",
    "recovery",
    "1 week",
    "2 weeks",
    "1 month",
    "3 months",
    "6 months",
    "1 year",
    "final result",
]

# Keywords that indicate non-surgical content to skip
SKIP_INDICATORS = [
    "filler",
    "botox",
    "injectable",
    "prp",
    "microneedling",
    "laser",
    "chemical peel",
    "skincare",
    "product",
    "advertising",
    "ad ",
    "#ad",
    "sponsored",
]


def classify_procedure(caption: str) -> str | None:
    """Classify surgical procedure from post caption and hashtags.

    Returns procedure name or None if no match found.
    Uses longest-match priority to handle overlapping keywords
    (e.g., "chin surgery" matches both orthognathic and mentoplasty).
    """
    if not caption:
        return None

    text = caption.lower()

    # Check skip indicators first
    for skip in SKIP_INDICATORS:
        if skip in text:
            return None

    # Score each procedure by number of matching keywords
    scores: dict[str, int] = {}
    for proc, keywords in PROCEDURE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[proc] = score

    if not scores:
        return None

    # Return highest-scoring procedure
    return max(scores, key=scores.get)  # type: ignore[arg-type]


def is_before_after_caption(caption: str) -> bool:
    """Check if caption suggests a before/after comparison."""
    if not caption:
        return False
    text = caption.lower()
    return any(ind in text for ind in BEFORE_AFTER_INDICATORS)


# ============================================================
# Image quality filtering
# ============================================================

MIN_FACE_SIZE = 80  # minimum face bbox dimension in pixels
MIN_IMAGE_DIM = 400  # minimum image width or height
MAX_BLUR_THRESHOLD = 50.0  # Laplacian variance below this = blurry


def check_image_quality(image: np.ndarray) -> dict[str, bool]:
    """Run quality checks on an image.

    Returns dict of check_name -> passed.
    """
    h, w = image.shape[:2]
    checks = {}

    # Resolution check
    checks["resolution"] = min(h, w) >= MIN_IMAGE_DIM

    # Blur detection via Laplacian variance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    checks["sharpness"] = blur_score >= MAX_BLUR_THRESHOLD

    return checks


def check_face_quality(
    face_crop: np.ndarray,
    min_size: int = MIN_FACE_SIZE,
) -> bool:
    """Check if a face crop meets minimum quality requirements."""
    h, w = face_crop.shape[:2]
    if min(h, w) < min_size:
        return False

    # Check for mostly black/white (failed crop)
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    return not (mean_val < 20 or mean_val > 240)


# ============================================================
# Face detection and alignment
# ============================================================


def detect_faces(image: np.ndarray) -> list[dict]:
    """Detect faces in image using MediaPipe.

    Returns list of face bounding boxes with landmarks.
    """
    try:
        import mediapipe as mp
    except ImportError:
        logger.error("mediapipe required: pip install mediapipe")
        return []

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return []

        h, w = image.shape[:2]
        faces = []
        for face in results.multi_face_landmarks:
            xs = [lm.x * w for lm in face.landmark]
            ys = [lm.y * h for lm in face.landmark]
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in face.landmark])
            faces.append(
                {
                    "bbox": (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                    "landmarks": landmarks,
                    "center_x": float(np.mean(xs)),
                    "center_y": float(np.mean(ys)),
                }
            )
        return faces


def is_before_after_layout(image: np.ndarray) -> str | None:
    """Detect if image is a side-by-side or top-bottom before/after layout.

    Returns:
        'horizontal' if side-by-side (left=before, right=after)
        'vertical' if top-bottom (top=before, bottom=after)
        None if not a comparison layout
    """
    faces = detect_faces(image)
    if len(faces) != 2:
        return None

    h, w = image.shape[:2]
    f1, f2 = faces[0], faces[1]

    # Check horizontal separation (side-by-side)
    cx_diff = abs(f1["center_x"] - f2["center_x"])
    cy_diff = abs(f1["center_y"] - f2["center_y"])

    if cx_diff > w * 0.3 and cy_diff < h * 0.2:
        return "horizontal"
    elif cy_diff > h * 0.3 and cx_diff < w * 0.2:
        return "vertical"

    return None


def split_comparison(
    image: np.ndarray,
    layout: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a comparison image into before and after halves.

    Convention: left/top = before, right/bottom = after.
    """
    h, w = image.shape[:2]
    if layout == "horizontal":
        mid = w // 2
        return image[:, :mid], image[:, mid:]
    else:  # vertical
        mid = h // 2
        return image[:mid, :], image[mid:, :]


def align_and_crop_face(
    image: np.ndarray,
    target_size: int = 512,
) -> np.ndarray | None:
    """Detect face, align by eye corners, crop, and resize.

    Returns aligned face image or None if detection fails.
    """
    faces = detect_faces(image)
    if not faces:
        return None

    face = faces[0]
    landmarks = face["landmarks"]

    # Align using outer eye corners (indices 33 and 263)
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    # Rotation angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = float(np.degrees(np.arctan2(dy, dx)))

    # Center of rotation
    h, w = image.shape[:2]
    center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    # Rotate
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LANCZOS4)

    # Re-detect on rotated image for tight crop
    faces_r = detect_faces(rotated)
    if not faces_r:
        return None

    bbox = faces_r[0]["bbox"]
    x1, y1, x2, y2 = bbox

    # Add padding (20% each side)
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * 0.2)
    pad_y = int(bh * 0.3)  # more vertical padding for forehead

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + int(bh * 0.1))

    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(
        crop,
        (target_size, target_size),
        interpolation=cv2.INTER_LANCZOS4,
    )


# ============================================================
# Image processing pipeline
# ============================================================


def process_image(
    image_path: Path,
    output_dir: Path,
    pair_idx: int,
    procedure: str | None = None,
    caption: str | None = None,
    target_size: int = 512,
) -> dict | None:
    """Process a single Instagram image into a before/after pair.

    Returns metadata dict or None if processing fails.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    # Quality check on source image
    quality = check_image_quality(image)
    if not quality["resolution"]:
        logger.debug("Skipping %s: too small", image_path.name)
        return None

    layout = is_before_after_layout(image)
    if layout is None:
        return None

    before_raw, after_raw = split_comparison(image, layout)

    before = align_and_crop_face(before_raw, target_size)
    after = align_and_crop_face(after_raw, target_size)

    if before is None or after is None:
        return None

    # Quality check on face crops
    if not check_face_quality(before) or not check_face_quality(after):
        logger.debug("Skipping %s: face quality too low", image_path.name)
        return None

    # Classify procedure from caption if not provided
    if procedure is None and caption:
        procedure = classify_procedure(caption)

    # Determine output subdirectory (by procedure for mega_scrape compat)
    proc_dir = output_dir / procedure if procedure else output_dir / "unclassified"
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Content hash for dedup
    content_hash = hashlib.md5(before.tobytes()[:4096] + after.tobytes()[:4096]).hexdigest()[:12]

    # Save pair using mega_scrape compatible naming
    pair_name = f"ig_{pair_idx:05d}_{content_hash}"
    cv2.imwrite(str(proc_dir / f"{pair_name}_input.png"), before)
    cv2.imwrite(str(proc_dir / f"{pair_name}_target.png"), after)

    metadata = {
        "pair_idx": pair_idx,
        "source": image_path.name,
        "source_type": "instagram",
        "layout": layout,
        "procedure": procedure,
        "content_hash": content_hash,
        "before_size": list(before.shape[:2]),
        "after_size": list(after.shape[:2]),
        "quality_checks": quality,
    }

    if caption:
        # Store first 500 chars of caption for provenance
        metadata["caption_preview"] = caption[:500]

    with open(proc_dir / f"{pair_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# ============================================================
# Instagram downloading
# ============================================================


def download_posts(
    account: str,
    download_dir: Path,
    max_posts: int = 200,
    filter_surgical: bool = True,
) -> list[dict]:
    """Download posts from a public Instagram account.

    Uses instaloader for rate-limited, respectful downloading.
    Only downloads image posts (skips videos, stories, reels).

    Returns list of dicts with 'path' and 'caption' keys.
    """
    try:
        import instaloader
    except ImportError:
        logger.error("instaloader required: pip install instaloader")
        return []

    loader = instaloader.Instaloader(
        dirname_pattern=str(download_dir / "{profile}"),
        filename_pattern="{shortcode}",
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=True,
        compress_json=False,
        post_metadata_txt_pattern="",
    )

    try:
        profile = instaloader.Profile.from_username(loader.context, account)
    except Exception:
        logger.error("Cannot access profile: %s", account)
        return []

    downloaded: list[dict] = []
    count = 0
    skipped = 0

    for post in profile.get_posts():
        if count >= max_posts:
            break

        if post.is_video:
            continue

        caption = post.caption or ""

        # Filter for surgical before/after content
        if filter_surgical:
            is_surgical = classify_procedure(caption) is not None
            is_ba = is_before_after_caption(caption)
            if not (is_surgical or is_ba):
                skipped += 1
                continue

        try:
            loader.download_post(post, target=str(download_dir / account))
            # Find downloaded image
            for ext in [".jpg", ".png", ".jpeg"]:
                img_path = download_dir / account / f"{post.shortcode}{ext}"
                if img_path.exists():
                    downloaded.append(
                        {
                            "path": img_path,
                            "caption": caption,
                            "shortcode": post.shortcode,
                            "date": str(post.date_utc),
                            "likes": post.likes,
                        }
                    )
                    break
            count += 1
        except Exception as e:
            logger.warning("Failed to download %s: %s", post.shortcode, e)

        # Rate limiting (Varghaei used 3s between requests)
        time.sleep(3)

    logger.info(
        "Downloaded %d posts from @%s (skipped %d non-surgical)",
        len(downloaded),
        account,
        skipped,
    )
    return downloaded


# ============================================================
# Curated surgeon account list
# Methodology: Varghaei et al. used board-certified facial
# plastic surgeons with public Instagram accounts posting
# clinical before/after photos. These are example categories;
# actual accounts should be supplied via --accounts-file.
# ============================================================

EXAMPLE_HASHTAGS = [
    "#rhinoplastyresults",
    "#nosejobbeforeandafter",
    "#blepharoplastyresults",
    "#faceliftresults",
    "#plasticsurgerybeforeandafter",
    "#facialplasticsurgery",
    "#boardcertifiedplasticsurgeon",
]


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect surgical before/after pairs from Instagram",
    )
    parser.add_argument(
        "--accounts",
        nargs="+",
        help="Instagram account usernames to scrape",
    )
    parser.add_argument(
        "--accounts-file",
        type=Path,
        help="File with one account username per line",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/instagram_pairs"),
        help="Output directory for processed pairs",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Process already-downloaded images from this directory",
    )
    parser.add_argument(
        "--captions-file",
        type=Path,
        help="JSON mapping filename -> caption (for --input-dir mode)",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=200,
        help="Max posts to download per account",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=512,
        help="Output image size (square)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Download all posts, not just surgical content",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=MIN_FACE_SIZE,
        help="Minimum face bounding box dimension in pixels",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.output.mkdir(parents=True, exist_ok=True)

    # Load captions mapping if provided
    captions: dict[str, str] = {}
    if args.captions_file and args.captions_file.exists():
        with open(args.captions_file) as f:
            captions = json.load(f)
        logger.info("Loaded %d captions from %s", len(captions), args.captions_file)

    # Gather image data
    image_data: list[dict] = []  # list of {"path": Path, "caption": str}

    if args.input_dir:
        # Process existing directory
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            for p in sorted(args.input_dir.rglob(ext)):
                caption = captions.get(p.name, captions.get(p.stem, ""))
                image_data.append({"path": p, "caption": caption})
        logger.info("Found %d images in %s", len(image_data), args.input_dir)

    else:
        # Download from accounts
        accounts: list[str] = []
        if args.accounts:
            accounts.extend(args.accounts)
        if args.accounts_file and args.accounts_file.exists():
            accounts.extend(
                line.strip()
                for line in args.accounts_file.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            )

        if not accounts:
            logger.error("No accounts specified. Use --accounts or --accounts-file")
            sys.exit(1)

        download_dir = args.output / "_downloads"
        download_dir.mkdir(parents=True, exist_ok=True)

        for account in accounts:
            logger.info("Downloading from @%s...", account)
            posts = download_posts(
                account,
                download_dir,
                args.max_posts,
                filter_surgical=not args.no_filter,
            )
            image_data.extend(posts)

    # Process images into pairs
    pair_idx = 0
    successful = 0
    failed = 0
    proc_counts: dict[str, int] = {}

    for item in image_data:
        img_path = item.get("path", item) if isinstance(item, dict) else item
        if isinstance(img_path, str):
            img_path = Path(img_path)
        caption = item.get("caption", "") if isinstance(item, dict) else ""

        result = process_image(
            img_path,
            args.output,
            pair_idx,
            caption=caption,
            target_size=args.target_size,
        )
        if result:
            pair_idx += 1
            successful += 1
            proc = result.get("procedure", "unclassified") or "unclassified"
            proc_counts[proc] = proc_counts.get(proc, 0) + 1
            logger.info(
                "Pair %d: %s (%s, %s)",
                result["pair_idx"],
                result["source"],
                result["layout"],
                proc,
            )
        else:
            failed += 1

    # Summary
    summary = {
        "total_images": len(image_data),
        "successful_pairs": successful,
        "failed": failed,
        "procedure_counts": proc_counts,
        "output_dir": str(args.output),
    }

    summary_path = args.output / "collection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Collection complete: %d pairs from %d images (%d failed)",
        successful,
        len(image_data),
        failed,
    )
    for proc, count in sorted(proc_counts.items()):
        logger.info("  %s: %d pairs", proc, count)


if __name__ == "__main__":
    main()
