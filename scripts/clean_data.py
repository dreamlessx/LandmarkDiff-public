"""Data cleaning pipeline: scraped images → verified faces → training pairs.

Integrates the neural face verifier into the data pipeline:
1. Scan scraped image directories for face quality issues
2. Restore fixable images (CodeFormer/GFPGAN/Real-ESRGAN)
3. Reject unusable images (too distorted, no face, wrong content)
4. Verify identity preservation after restoration
5. Deduplicate near-identical images using ArcFace embeddings
6. Output clean, verified face dataset ready for pair generation

Usage:
    # Clean a single procedure's scraped images
    python scripts/clean_data.py data/scraped/rhinoplasty/ --output data/clean/rhinoplasty/

    # Clean all procedures
    python scripts/clean_data.py data/scraped/ --all-procedures --output data/clean/

    # Dedup + clean
    python scripts/clean_data.py data/scraped/ --all-procedures --dedup --output data/clean/

SLURM:
    sbatch --partition=batch_gpu --gres=gpu:1 --mem=32G --time=8:00:00 \
           --wrap="python scripts/clean_data.py data/scraped/ --all-procedures --dedup --output data/clean/"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.face_verifier import (
    analyze_distortions,
    get_face_embedding,
    verify_and_restore,
)
from landmarkdiff.landmarks import extract_landmarks

# ---------------------------------------------------------------------------
# Face validation (is this actually a usable face photo?)
# ---------------------------------------------------------------------------


def validate_face_image(image: np.ndarray) -> dict:
    """Check if image contains a valid, usable face for training.

    Rejects:
    - No face detected
    - Multiple faces (ambiguous — which is the patient?)
    - Face too small (<100px inter-ocular distance)
    - Face occluded (missing key landmarks)
    - Non-face images that slipped through scraping
    """
    result = {
        "valid": False,
        "reason": "",
        "face_count": 0,
        "iod": 0.0,
        "confidence": 0.0,
    }

    # Check basic image properties
    if image is None or image.size == 0:
        result["reason"] = "empty_image"
        return result

    h, w = image.shape[:2]
    if h < 64 or w < 64:
        result["reason"] = "too_small"
        return result

    # Detect face
    face = extract_landmarks(image)
    if face is None:
        result["reason"] = "no_face"
        return result

    result["face_count"] = 1  # MediaPipe returns strongest face
    result["confidence"] = float(face.confidence)

    # Check confidence
    if face.confidence < 0.7:
        result["reason"] = "low_confidence"
        return result

    # Check inter-ocular distance (face size proxy)
    coords = face.pixel_coords
    left_eye = coords[33]
    right_eye = coords[263]
    iod = float(np.linalg.norm(left_eye - right_eye))
    result["iod"] = iod

    if iod < 30:
        result["reason"] = "face_too_small"
        return result

    # Check key landmarks are present and reasonable
    coords[1]
    chin = coords[152]
    forehead = coords[10]

    face_height = float(np.linalg.norm(forehead - chin))
    if face_height < 50:
        result["reason"] = "face_too_small"
        return result

    # Check face is within image bounds (not cropped off)
    margin = 10
    if (
        left_eye[0] < margin
        or right_eye[0] > w - margin
        or forehead[1] < margin
        or chin[1] > h - margin
    ):
        result["reason"] = "face_cropped"
        return result

    result["valid"] = True
    result["reason"] = "ok"
    return result


# ---------------------------------------------------------------------------
# Deduplication using ArcFace embeddings
# ---------------------------------------------------------------------------


def deduplicate_faces(
    image_paths: list[Path],
    similarity_threshold: float = 0.85,
) -> tuple[list[Path], list[Path]]:
    """Remove near-duplicate faces using ArcFace embedding similarity.

    Groups images by identity, keeps the highest quality image from each
    group. Two images with ArcFace cosine similarity > threshold are
    considered duplicates of the same person/angle.

    Returns (unique_paths, duplicate_paths).
    """
    print(f"Deduplicating {len(image_paths)} images (threshold={similarity_threshold})...")

    embeddings = []
    valid_paths = []

    for i, path in enumerate(image_paths):
        if (i + 1) % 100 == 0:
            print(f"  Embedding {i + 1}/{len(image_paths)}...")

        image = cv2.imread(str(path))
        if image is None:
            continue

        image = cv2.resize(image, (512, 512))
        emb = get_face_embedding(image)
        if emb is not None:
            embeddings.append(emb)
            valid_paths.append(path)

    if not embeddings:
        return image_paths, []

    # Build similarity matrix (O(n²) but fine for <100K images)
    emb_matrix = np.stack(embeddings)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
    emb_normed = emb_matrix / norms

    # Find clusters of duplicates using greedy approach
    n = len(valid_paths)
    used = set()
    unique = []
    duplicates = []

    for i in range(n):
        if i in used:
            continue

        # Find all images similar to this one
        sims = emb_normed[i] @ emb_normed.T
        cluster = [j for j in range(n) if j not in used and sims[j] > similarity_threshold]

        if len(cluster) <= 1:
            unique.append(valid_paths[i])
            used.add(i)
        else:
            # Keep the one with best quality score
            best_idx = cluster[0]
            best_quality = 0.0
            for j in cluster:
                img = cv2.imread(str(valid_paths[j]))
                if img is not None:
                    img = cv2.resize(img, (512, 512))
                    report = analyze_distortions(img)
                    if report.quality_score > best_quality:
                        best_quality = report.quality_score
                        best_idx = j

            unique.append(valid_paths[best_idx])
            for j in cluster:
                used.add(j)
                if j != best_idx:
                    duplicates.append(valid_paths[j])

    # Add paths where embeddings failed (keep them — can't dedup)
    no_emb_paths = set(image_paths) - set(valid_paths)
    unique.extend(no_emb_paths)

    print(f"  Unique: {len(unique)} | Duplicates removed: {len(duplicates)}")
    return unique, duplicates


# ---------------------------------------------------------------------------
# Full cleaning pipeline
# ---------------------------------------------------------------------------


def clean_directory(
    input_dir: str,
    output_dir: str,
    quality_threshold: float = 55.0,
    identity_threshold: float = 0.6,
    restore_mode: str = "auto",
    dedup: bool = False,
    dedup_threshold: float = 0.85,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
) -> dict:
    """Full data cleaning pipeline for a directory of scraped face images.

    Pipeline:
    1. Validate each image (face detected? usable quality?)
    2. Analyze distortions (blur, noise, filters, etc.)
    3. Restore fixable images with neural nets
    4. Verify identity preserved after restoration
    5. Optionally deduplicate by ArcFace embedding similarity
    6. Output clean dataset + comprehensive report

    Args:
        input_dir: Directory of raw scraped images.
        output_dir: Where to save cleaned images.
        quality_threshold: Min quality score to pass without restoration.
        identity_threshold: Min ArcFace similarity after restoration.
        restore_mode: Neural restoration mode.
        dedup: Whether to deduplicate by face identity.
        dedup_threshold: ArcFace similarity to consider as duplicate.

    Returns:
        Dict with cleaning statistics.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    # Create output structure
    clean_dir = out_path / "clean"
    restored_dir = out_path / "restored"
    rejected_dir = out_path / "rejected"
    report_dir = out_path / "reports"
    clean_dir.mkdir(parents=True, exist_ok=True)
    restored_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image files
    image_files = sorted(
        [f for f in in_path.iterdir() if f.suffix.lower() in extensions and f.is_file()]
    )

    stats = {
        "total": len(image_files),
        "valid_face": 0,
        "invalid_face": 0,
        "passed_clean": 0,
        "restored": 0,
        "rejected_quality": 0,
        "rejected_identity": 0,
        "duplicates_removed": 0,
        "final_count": 0,
        "rejection_reasons": defaultdict(int),
        "distortion_types": defaultdict(int),
        "avg_quality_before": [],
        "avg_quality_after": [],
    }

    print(f"\n{'=' * 60}")
    print(f"Cleaning: {in_path}")
    print(f"Images found: {len(image_files)}")
    print(f"Quality threshold: {quality_threshold}")
    print(f"{'=' * 60}\n")

    # Phase 1: Validate + Restore
    clean_paths = []

    for i, img_file in enumerate(image_files):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"[{i + 1}/{len(image_files)}] Processing {img_file.name}...")

        image = cv2.imread(str(img_file))
        if image is None:
            stats["rejection_reasons"]["unreadable"] += 1
            continue

        image = cv2.resize(image, (512, 512))

        # Step 1: Is this a valid face?
        validation = validate_face_image(image)
        if not validation["valid"]:
            stats["invalid_face"] += 1
            stats["rejection_reasons"][validation["reason"]] += 1
            cv2.imwrite(str(rejected_dir / img_file.name), image)
            continue

        stats["valid_face"] += 1

        # Step 2: Quality analysis + restoration
        result = verify_and_restore(
            image,
            quality_threshold=quality_threshold,
            identity_threshold=identity_threshold,
            restore_mode=restore_mode,
        )

        stats["avg_quality_before"].append(result.distortion_report.quality_score)
        stats["distortion_types"][result.distortion_report.primary_distortion] += 1

        if "rejected" in result.restoration_stages:
            stats["rejected_quality"] += 1
            stats["rejection_reasons"]["quality_too_low"] += 1
            cv2.imwrite(str(rejected_dir / img_file.name), image)
            continue

        if not result.restoration_stages:
            # Passed clean — no restoration needed
            stats["passed_clean"] += 1
            stats["avg_quality_after"].append(result.post_quality_score)
            out_file = clean_dir / img_file.name
            cv2.imwrite(str(out_file), image)
            clean_paths.append(out_file)
        else:
            # Was restored
            if result.identity_preserved:
                stats["restored"] += 1
                stats["avg_quality_after"].append(result.post_quality_score)
                out_file = restored_dir / img_file.name
                cv2.imwrite(str(out_file), result.restored)
                clean_paths.append(out_file)
            else:
                stats["rejected_identity"] += 1
                stats["rejection_reasons"]["identity_drift"] += 1
                cv2.imwrite(str(rejected_dir / img_file.name), image)

    # Phase 2: Deduplication
    if dedup and clean_paths:
        unique_paths, dup_paths = deduplicate_faces(clean_paths, dedup_threshold)
        stats["duplicates_removed"] = len(dup_paths)

        # Move duplicates to rejected
        for dup in dup_paths:
            dest = rejected_dir / f"dup_{dup.name}"
            shutil.move(str(dup), str(dest))

        clean_paths = unique_paths

    stats["final_count"] = len(clean_paths)

    # Compute averages
    stats["avg_quality_before"] = (
        float(np.mean(stats["avg_quality_before"])) if stats["avg_quality_before"] else 0.0
    )
    stats["avg_quality_after"] = (
        float(np.mean(stats["avg_quality_after"])) if stats["avg_quality_after"] else 0.0
    )

    # Convert defaultdicts for JSON
    stats["rejection_reasons"] = dict(stats["rejection_reasons"])
    stats["distortion_types"] = dict(stats["distortion_types"])

    # Save report
    report_text = _format_report(stats, in_path.name)
    (report_dir / "cleaning_report.txt").write_text(report_text)
    with open(report_dir / "cleaning_report.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{report_text}")
    print(f"\nClean images: {clean_dir}/")
    print(f"Restored images: {restored_dir}/")
    print(f"Rejected images: {rejected_dir}/")
    print(f"Reports: {report_dir}/")

    return stats


def _format_report(stats: dict, name: str) -> str:
    """Format cleaning statistics into readable report."""
    lines = [
        f"Data Cleaning Report: {name}",
        "=" * 50,
        f"Total input images:      {stats['total']}",
        f"Valid faces detected:     {stats['valid_face']}",
        f"Invalid/no face:          {stats['invalid_face']}",
        "",
        "Processing Results:",
        f"  Passed clean:           {stats['passed_clean']}",
        f"  Restored (neural net):  {stats['restored']}",
        f"  Rejected (quality):     {stats['rejected_quality']}",
        f"  Rejected (identity):    {stats['rejected_identity']}",
        f"  Duplicates removed:     {stats['duplicates_removed']}",
        "",
        f"Final clean count:        {stats['final_count']}",
        f"Yield rate:               {stats['final_count'] / max(stats['total'], 1) * 100:.1f}%",
        "",
        f"Avg quality before:       {stats['avg_quality_before']:.1f}/100",
        f"Avg quality after:        {stats['avg_quality_after']:.1f}/100",
        "",
        "Rejection Reasons:",
    ]
    for reason, count in sorted(stats["rejection_reasons"].items(), key=lambda x: -x[1]):
        lines.append(f"  {reason}: {count}")

    lines.append("")
    lines.append("Distortion Types Detected:")
    for dtype, count in sorted(stats["distortion_types"].items(), key=lambda x: -x[1]):
        lines.append(f"  {dtype}: {count}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Clean scraped face images for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input directory of scraped images")
    parser.add_argument("--output", "-o", help="Output directory for clean images")
    parser.add_argument(
        "--all-procedures", action="store_true", help="Process all procedure subdirectories"
    )
    parser.add_argument(
        "--dedup", action="store_true", help="Remove near-duplicate faces by ArcFace similarity"
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="ArcFace similarity threshold for dedup (default: 0.85)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=55.0,
        help="Min quality to pass without restoration (default: 55)",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=0.6,
        help="Min identity similarity after restoration (default: 0.6)",
    )
    parser.add_argument(
        "--restore-mode",
        default="auto",
        choices=["auto", "codeformer", "gfpgan", "all", "none"],
        help="Restoration mode (default: auto)",
    )

    args = parser.parse_args()
    in_path = Path(args.input)

    if args.all_procedures:
        # Process each procedure subdirectory
        procedures = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
        all_stats = {}

        for proc in procedures:
            proc_dir = in_path / proc
            if not proc_dir.exists():
                print(f"Skipping {proc} (directory not found)")
                continue

            out_dir = args.output or str(in_path.parent / f"{in_path.name}_clean")
            proc_out = str(Path(out_dir) / proc)

            stats = clean_directory(
                str(proc_dir),
                proc_out,
                quality_threshold=args.quality_threshold,
                identity_threshold=args.identity_threshold,
                restore_mode=args.restore_mode,
                dedup=args.dedup,
                dedup_threshold=args.dedup_threshold,
            )
            all_stats[proc] = stats

        # Summary across all procedures
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        total_in = sum(s["total"] for s in all_stats.values())
        total_out = sum(s["final_count"] for s in all_stats.values())
        for proc, stats in all_stats.items():
            print(
                f"  {proc}: {stats['total']} → {stats['final_count']} "
                f"({stats['final_count'] / max(stats['total'], 1) * 100:.0f}% yield)"
            )
        print(
            f"  TOTAL: {total_in} → {total_out} ({total_out / max(total_in, 1) * 100:.0f}% yield)"
        )

    else:
        out_dir = args.output or str(in_path.parent / f"{in_path.name}_clean")
        clean_directory(
            str(in_path),
            out_dir,
            quality_threshold=args.quality_threshold,
            identity_threshold=args.identity_threshold,
            restore_mode=args.restore_mode,
            dedup=args.dedup,
            dedup_threshold=args.dedup_threshold,
        )


if __name__ == "__main__":
    main()
