#!/usr/bin/env python3
"""Process HDA Plastic Surgery Face Database for LandmarkDiff training.

Extracts MediaPipe landmarks, computes displacement vectors, generates
conditioning maps and surgical masks, and outputs training-ready pairs
in the standard LandmarkDiff format.

Database: Rathgeb et al., "Plastic Surgery: An Obstacle for Deep Face
Recognition?", CVPRW 2020.

Input structure:
    data/plastic_surgery_db/
    ├── Eyebrow/    {id}_b.jpg (before), {id}_a.jpg (after)
    ├── Eyelid/     ...
    ├── Facelift/   ...
    ├── FacialBones/...
    └── Nose/       ...

Output structure:
    data/hda_processed/
    ├── {procedure}_{id}_input.png
    ├── {procedure}_{id}_target.png
    ├── {procedure}_{id}_conditioning.png
    ├── {procedure}_{id}_mask.png
    ├── {procedure}_{id}_canny.png
    ├── metadata.json
    └── displacement_model.npz

Usage:
    python scripts/process_hda_database.py
    python scripts/process_hda_database.py --resolution 512 --min-quality 0.3
    python scripts/process_hda_database.py \\
        --db-path data/plastic_surgery_db --output data/hda_processed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from landmarkdiff.conditioning import generate_conditioning
from landmarkdiff.displacement_model import (
    DisplacementModel,
    _compute_alignment_quality,
    _normalized_coords_2d,
    classify_procedure,
)
from landmarkdiff.landmarks import FaceLandmarks, extract_landmarks
from landmarkdiff.masking import MASK_CONFIG, generate_surgical_mask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# HDA directory name → LandmarkDiff procedure mapping
HDA_PROCEDURE_MAP = {
    "Nose": "rhinoplasty",
    "Eyelid": "blepharoplasty",
    "Facelift": "rhytidectomy",
    "FacialBones": "orthognathic",
    "Eyebrow": "blepharoplasty",  # Eyebrow lifts are blepharoplasty-adjacent
}


def discover_pairs(db_path: Path) -> list[dict]:
    """Discover before/after image pairs from the HDA database.

    HDA naming convention: {id}_b.jpg (before), {id}_a.jpg (after)

    Returns:
        List of dicts with keys: id, before_path, after_path, hda_category, procedure
    """
    pairs = []

    for category_dir in sorted(db_path.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        if category not in HDA_PROCEDURE_MAP:
            logger.warning("Skipping unknown category: %s", category)
            continue

        procedure = HDA_PROCEDURE_MAP[category]

        # Find all _b.jpg files and match with _a.jpg
        before_files = sorted(category_dir.glob("*_b.jpg"))
        for before_path in before_files:
            pair_id = before_path.stem.replace("_b", "")
            after_path = category_dir / f"{pair_id}_a.jpg"

            if not after_path.exists():
                # Try other extensions
                for ext in [".jpeg", ".png", ".bmp"]:
                    alt = category_dir / f"{pair_id}_a{ext}"
                    if alt.exists():
                        after_path = alt
                        break

            if not after_path.exists():
                logger.warning("No after image for %s/%s", category, pair_id)
                continue

            pairs.append(
                {
                    "id": pair_id,
                    "before_path": before_path,
                    "after_path": after_path,
                    "hda_category": category,
                    "procedure": procedure,
                }
            )

    return pairs


def process_pair(
    pair: dict,
    output_dir: Path,
    resolution: int = 512,
    min_quality: float = 0.0,
    min_confidence: float = 0.3,
) -> dict | None:
    """Process a single before/after pair.

    Returns metadata dict on success, None on failure.
    """
    pair_id = pair["id"]
    procedure = pair["procedure"]
    category = pair["hda_category"]
    prefix = f"{procedure}_{category}_{pair_id}"

    # Load images
    before_img = cv2.imread(str(pair["before_path"]))
    after_img = cv2.imread(str(pair["after_path"]))

    if before_img is None:
        logger.warning("[%s] Failed to load before image", prefix)
        return None
    if after_img is None:
        logger.warning("[%s] Failed to load after image", prefix)
        return None

    # Extract landmarks
    face_before = extract_landmarks(before_img, min_detection_confidence=min_confidence)
    if face_before is None:
        logger.warning("[%s] No face detected in before image", prefix)
        return None

    face_after = extract_landmarks(after_img, min_detection_confidence=min_confidence)
    if face_after is None:
        logger.warning("[%s] No face detected in after image", prefix)
        return None

    # Compute displacements
    coords_before = _normalized_coords_2d(face_before)
    coords_after = _normalized_coords_2d(face_after)
    displacements = coords_after - coords_before
    magnitudes = np.linalg.norm(displacements, axis=1)

    # Alignment quality
    quality = _compute_alignment_quality(coords_before, coords_after)
    if quality < min_quality:
        logger.info("[%s] Quality %.3f < threshold %.3f, skipping", prefix, quality, min_quality)
        return None

    # Auto-classify procedure from displacements
    auto_procedure = classify_procedure(displacements)

    # Resize both images to target resolution
    before_resized = cv2.resize(before_img, (resolution, resolution))
    after_resized = cv2.resize(after_img, (resolution, resolution))

    # Generate conditioning from the AFTER landmarks (target face)
    # Scale landmarks to the new resolution
    face_after_scaled = FaceLandmarks(
        landmarks=face_after.landmarks,
        image_width=resolution,
        image_height=resolution,
        confidence=face_after.confidence,
    )

    landmark_img, canny, wireframe = generate_conditioning(
        face_after_scaled, resolution, resolution
    )

    # Generate surgical mask using the BEFORE landmarks + known procedure
    face_before_scaled = FaceLandmarks(
        landmarks=face_before.landmarks,
        image_width=resolution,
        image_height=resolution,
        confidence=face_before.confidence,
    )

    # Use the HDA category-mapped procedure for mask generation
    mask_procedure = procedure
    if mask_procedure not in MASK_CONFIG:
        mask_procedure = "rhinoplasty"  # Safe fallback

    mask = generate_surgical_mask(face_before_scaled, mask_procedure, resolution, resolution)

    # Save all outputs
    cv2.imwrite(str(output_dir / f"{prefix}_input.png"), before_resized)
    cv2.imwrite(str(output_dir / f"{prefix}_target.png"), after_resized)
    cv2.imwrite(str(output_dir / f"{prefix}_conditioning.png"), landmark_img)
    cv2.imwrite(str(output_dir / f"{prefix}_mask.png"), (mask * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / f"{prefix}_canny.png"), canny)

    return {
        "prefix": prefix,
        "pair_id": pair_id,
        "hda_category": category,
        "procedure": procedure,
        "auto_procedure": auto_procedure,
        "quality_score": float(quality),
        "mean_displacement": float(np.mean(magnitudes)),
        "max_displacement": float(np.max(magnitudes)),
        "before_resolution": [before_img.shape[1], before_img.shape[0]],
        "after_resolution": [after_img.shape[1], after_img.shape[0]],
        "landmarks_before": coords_before.tolist(),
        "landmarks_after": coords_after.tolist(),
        "displacements": displacements.tolist(),
        "source": "HDA_PlasticSurgery_CVPRW2020",
    }


def main():
    parser = argparse.ArgumentParser(description="Process HDA Plastic Surgery Database")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=ROOT / "data" / "plastic_surgery_db",
        help="Path to HDA database root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "hda_processed",
        help="Output directory for processed pairs",
    )
    parser.add_argument("--resolution", type=int, default=512, help="Target resolution")
    parser.add_argument("--min-quality", type=float, default=0.2, help="Min alignment quality")
    parser.add_argument(
        "--min-confidence", type=float, default=0.3, help="Min face detection confidence"
    )
    parser.add_argument(
        "--fit-model", action="store_true", default=True, help="Fit displacement model"
    )
    parser.add_argument("--no-fit-model", dest="fit_model", action="store_false")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("HDA Plastic Surgery Database Processor")
    logger.info("=" * 60)
    logger.info("Database: %s", args.db_path)
    logger.info("Output: %s", args.output)
    logger.info("Resolution: %d", args.resolution)
    logger.info("Min quality: %.2f", args.min_quality)

    # Discover pairs
    pairs = discover_pairs(args.db_path)
    logger.info("Found %d before/after pairs", len(pairs))

    # Count by category
    categories = {}
    for p in pairs:
        cat = p["hda_category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        logger.info("  %s: %d pairs → %s", cat, count, HDA_PROCEDURE_MAP.get(cat, "?"))

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Process pairs
    results = []
    displacement_data = []
    failed = 0
    start_time = time.time()

    for i, pair in enumerate(pairs):
        result = process_pair(
            pair,
            args.output,
            resolution=args.resolution,
            min_quality=args.min_quality,
            min_confidence=args.min_confidence,
        )

        if result is None:
            failed += 1
        else:
            results.append(result)
            # Collect displacement data for model fitting
            displacement_data.append(
                {
                    "displacements": np.array(result["displacements"]),
                    "procedure": result["procedure"],
                    "quality_score": result["quality_score"],
                }
            )

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(pairs) - i - 1) / max(rate, 0.01)
            logger.info(
                "Progress: %d/%d (%.0f%%) | %d ok, %d failed | %.1f pairs/s | ETA %.0fs",
                i + 1,
                len(pairs),
                100 * (i + 1) / len(pairs),
                len(results),
                failed,
                rate,
                eta,
            )

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("Processing Complete")
    logger.info("=" * 60)
    logger.info("Total pairs: %d", len(pairs))
    logger.info("Successful: %d (%.1f%%)", len(results), 100 * len(results) / max(len(pairs), 1))
    logger.info("Failed: %d", failed)
    logger.info("Time: %.1fs (%.1f pairs/s)", elapsed, len(pairs) / max(elapsed, 0.01))

    # Procedure breakdown
    proc_counts = {}
    for r in results:
        proc = r["procedure"]
        proc_counts[proc] = proc_counts.get(proc, 0) + 1
    logger.info("")
    logger.info("Procedure breakdown:")
    for proc, count in sorted(proc_counts.items()):
        logger.info("  %s: %d pairs", proc, count)

    # Auto-classification accuracy
    correct = sum(1 for r in results if r["auto_procedure"] == r["procedure"])
    logger.info("")
    logger.info(
        "Auto-classification accuracy: %d/%d (%.1f%%)",
        correct,
        len(results),
        100 * correct / max(len(results), 1),
    )

    # Save metadata
    metadata = {
        "source": "HDA Plastic Surgery Face Database (Rathgeb et al., CVPRW 2020)",
        "citation": (
            "C. Rathgeb, P. Drozdowski, D. Fischer, C. Busch. "
            '"Plastic Surgery: An Obstacle for Deep Face Recognition?" '
            "IEEE/CVF CVPRW 2020."
        ),
        "license": "Non-commercial research use only",
        "total_pairs": len(results),
        "resolution": args.resolution,
        "min_quality": args.min_quality,
        "procedure_counts": proc_counts,
        "pairs": {r["prefix"]: r for r in results},
    }

    # Remove large numpy data from metadata (keep only scalar stats)
    for key in metadata["pairs"]:
        for field in ["landmarks_before", "landmarks_after", "displacements"]:
            if field in metadata["pairs"][key]:
                del metadata["pairs"][key][field]

    meta_path = args.output / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)

    # Fit and save displacement model
    if args.fit_model and displacement_data:
        logger.info("")
        logger.info("Fitting displacement model from %d pairs...", len(displacement_data))
        model = DisplacementModel()
        model.fit(displacement_data)

        model_path = args.output / "displacement_model.npz"
        model.save(model_path)

        # Print model summary
        summary = model.get_summary()
        for proc, stats in summary.get("procedures", {}).items():
            logger.info(
                "  %s: %d samples, mean_mag=%.5f, max_mag=%.5f",
                proc,
                stats["n_samples"],
                stats["global_mean_magnitude"],
                stats["global_max_magnitude"],
            )

        # Also save displacement stats as JSON for inspection
        stats_path = args.output / "displacement_stats.json"
        with open(stats_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved displacement stats to %s", stats_path)

    # Quality statistics
    if results:
        qualities = [r["quality_score"] for r in results]
        logger.info("")
        logger.info("Quality score distribution:")
        logger.info("  Mean: %.3f", np.mean(qualities))
        logger.info("  Median: %.3f", np.median(qualities))
        logger.info("  Min: %.3f", np.min(qualities))
        logger.info("  Max: %.3f", np.max(qualities))

        disps = [r["mean_displacement"] for r in results]
        logger.info("")
        logger.info("Mean displacement distribution:")
        logger.info("  Mean: %.5f", np.mean(disps))
        logger.info("  Median: %.5f", np.median(disps))
        logger.info("  Min: %.5f", np.min(disps))
        logger.info("  Max: %.5f", np.max(disps))


if __name__ == "__main__":
    main()
