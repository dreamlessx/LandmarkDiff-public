#!/usr/bin/env python3
"""Batch inference for surgical outcome prediction.

Processes a directory of face images through the full LandmarkDiff pipeline
(landmark extraction → deformation → ControlNet/TPS → neural post-processing)
and outputs before/after results with quality metrics.

Designed for HPC batch processing with SLURM array jobs.

Usage:
    # Single procedure, all images
    python scripts/batch_inference.py \
        --input data/celeba_hq_extracted \
        --output results/rhinoplasty \
        --procedure rhinoplasty --intensity 65

    # All procedures, neural post-processing
    python scripts/batch_inference.py \
        --input data/test_images \
        --output results/all_procedures \
        --all-procedures --neural

    # SLURM array job (processes subset)
    python scripts/batch_inference.py \
        --input data/test_images \
        --output results/batch \
        --procedure rhinoplasty \
        --array-index $SLURM_ARRAY_TASK_ID \
        --array-total 4

    # With data-driven displacement model
    python scripts/batch_inference.py \
        --input data/test_images \
        --output results/data_driven \
        --procedure rhinoplasty \
        --displacement-model data/displacement_model.npz
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.evaluation import (
    classify_fitzpatrick_ita,
    compute_lpips,
    compute_nme,
    compute_ssim,
)
from landmarkdiff.inference import mask_composite
from landmarkdiff.landmarks import extract_landmarks, render_landmark_image, visualize_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps

logger = logging.getLogger(__name__)

PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]


def process_image(
    image_path: Path,
    procedure: str,
    intensity: float,
    output_dir: Path,
    use_neural: bool = False,
    displacement_model_path: str | None = None,
    save_intermediates: bool = False,
) -> dict | None:
    """Process a single image through the full pipeline.

    Returns a results dict with metrics, or None if face detection fails.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    img_512 = cv2.resize(img, (512, 512))
    t0 = time.time()

    # Extract landmarks
    face = extract_landmarks(img_512)
    if face is None:
        return None

    # Deformation
    manip = apply_procedure_preset(
        face,
        procedure,
        intensity,
        image_size=512,
        displacement_model_path=displacement_model_path,
    )

    # Mask + conditioning
    mask = generate_surgical_mask(face, procedure, 512, 512)
    landmark_img = render_landmark_image(manip, 512, 512)

    # TPS warp + composite
    warped = warp_image_tps(img_512, face.pixel_coords, manip.pixel_coords)
    composited = mask_composite(warped, img_512, mask)

    # Neural post-processing
    if use_neural:
        try:
            from landmarkdiff.postprocess import full_postprocess

            pp = full_postprocess(
                generated=composited,
                original=img_512,
                mask=mask,
                restore_mode="codeformer",
                codeformer_fidelity=0.7,
                use_realesrgan=True,
                use_laplacian_blend=True,
                sharpen_strength=0.25,
                verify_identity=True,
                identity_threshold=0.6,
            )
            enhanced = pp["image"]
            identity_check = pp.get("identity_check", {})
        except Exception as e:
            enhanced = composited
            identity_check = {"error": str(e)}
    else:
        enhanced = composited
        identity_check = {}

    elapsed = time.time() - t0

    # Compute quality metrics
    ssim_val = compute_ssim(enhanced, img_512)
    lpips_val = compute_lpips(enhanced, img_512)
    nme_val = 0.0
    if face is not None and manip is not None:
        nme_val = compute_nme(manip.pixel_coords, face.pixel_coords)

    # Fitzpatrick type
    try:
        fitz = classify_fitzpatrick_ita(img_512)
    except Exception:
        fitz = "?"

    # Save outputs
    stem = image_path.stem
    proc_dir = output_dir / procedure
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Before/after pair
    before_after = np.hstack([img_512, enhanced])
    cv2.imwrite(str(proc_dir / f"{stem}_before_after.png"), before_after)

    # Enhanced image alone
    cv2.imwrite(str(proc_dir / f"{stem}_output.png"), enhanced)

    if save_intermediates:
        inter_dir = proc_dir / "intermediates"
        inter_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(inter_dir / f"{stem}_original.png"), img_512)
        cv2.imwrite(str(inter_dir / f"{stem}_warped.png"), warped)
        cv2.imwrite(str(inter_dir / f"{stem}_composited.png"), composited)
        cv2.imwrite(str(inter_dir / f"{stem}_mask.png"), (mask * 255).astype(np.uint8))
        cv2.imwrite(str(inter_dir / f"{stem}_conditioning.png"), landmark_img)
        cv2.imwrite(
            str(inter_dir / f"{stem}_landmarks.png"),
            visualize_landmarks(img_512, face, radius=2),
        )

    return {
        "image": stem,
        "procedure": procedure,
        "intensity": intensity,
        "fitzpatrick": fitz,
        "ssim": float(ssim_val),
        "lpips": float(lpips_val),
        "nme": float(nme_val),
        "identity_check": identity_check,
        "elapsed_seconds": round(elapsed, 2),
        "neural_postprocess": use_neural,
    }


def create_intensity_grid(
    image_path: Path,
    procedure: str,
    intensities: list[float],
    output_path: Path,
    displacement_model_path: str | None = None,
) -> bool:
    """Create a multi-intensity comparison grid for a single image.

    Shows the original plus outputs at each intensity level side-by-side.
    Returns True on success, False if face detection fails.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    img_512 = cv2.resize(img, (512, 512))
    face = extract_landmarks(img_512)
    if face is None:
        return False

    panels = [img_512.copy()]  # original first
    labels = ["Original"]

    for intensity in intensities:
        manip = apply_procedure_preset(
            face,
            procedure,
            intensity,
            image_size=512,
            displacement_model_path=displacement_model_path,
        )
        mask = generate_surgical_mask(face, procedure, 512, 512)
        warped = warp_image_tps(img_512, face.pixel_coords, manip.pixel_coords)
        composited = mask_composite(warped, img_512, mask)
        panels.append(composited)
        labels.append(f"{intensity:.0f}%")

    # Add labels to panels
    for _i, (panel, label) in enumerate(zip(panels, labels, strict=False)):
        cv2.putText(
            panel,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    grid = np.hstack(panels)
    cv2.imwrite(str(output_path), grid)
    return True


def collect_images(input_dir: Path) -> list[Path]:
    """Collect all image files from a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in extensions and f.is_file())
    return files


def main():
    parser = argparse.ArgumentParser(description="Batch LandmarkDiff inference")
    parser.add_argument("--input", required=True, help="Input directory of face images")
    parser.add_argument("--output", default="results/batch", help="Output directory")
    parser.add_argument("--procedure", default="rhinoplasty", choices=PROCEDURES)
    parser.add_argument(
        "--all-procedures", action="store_true", help="Run all 6 procedures for each image"
    )
    parser.add_argument("--intensity", type=float, default=65.0)
    parser.add_argument(
        "--neural",
        action="store_true",
        help="Enable neural post-processing (CodeFormer + Real-ESRGAN)",
    )
    parser.add_argument(
        "--save-intermediates", action="store_true", help="Save intermediate pipeline outputs"
    )
    parser.add_argument(
        "--displacement-model", default=None, help="Path to data-driven displacement model (.npz)"
    )
    parser.add_argument(
        "--max-images", type=int, default=None, help="Limit number of images to process"
    )
    parser.add_argument(
        "--array-index", type=int, default=None, help="SLURM array task index (0-based)"
    )
    parser.add_argument("--array-total", type=int, default=None, help="Total SLURM array tasks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--intensity-grid", action="store_true", help="Generate multi-intensity comparison grids"
    )
    parser.add_argument(
        "--grid-intensities",
        type=float,
        nargs="+",
        default=[25, 50, 75, 100],
        help="Intensity levels for grid mode",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    # Collect images
    all_images = collect_images(input_dir)
    if not all_images:
        logger.error("No images found in %s", input_dir)
        sys.exit(1)

    # SLURM array job partitioning
    if args.array_index is not None and args.array_total is not None:
        chunk_size = len(all_images) // args.array_total
        start = args.array_index * chunk_size
        end = start + chunk_size if args.array_index < args.array_total - 1 else len(all_images)
        all_images = all_images[start:end]
        logger.info(
            "SLURM array %d/%d: processing images %d-%d (%d images)",
            args.array_index,
            args.array_total,
            start,
            end,
            len(all_images),
        )

    if args.max_images:
        all_images = all_images[: args.max_images]

    procedures = PROCEDURES if args.all_procedures else [args.procedure]

    # Intensity grid mode
    if args.intensity_grid:
        logger.info("Generating intensity grids for %d images...", len(all_images))
        logger.info("Intensities: %s", args.grid_intensities)
        grid_dir = output_dir / "intensity_grids"
        grid_dir.mkdir(parents=True, exist_ok=True)
        success = 0
        for i, img_path in enumerate(all_images):
            for proc in procedures:
                out_path = grid_dir / f"{img_path.stem}_{proc}_grid.png"
                ok = create_intensity_grid(
                    img_path,
                    proc,
                    args.grid_intensities,
                    out_path,
                    displacement_model_path=args.displacement_model,
                )
                if ok:
                    success += 1
                    logger.info("[%d/%d] %s/%s: OK", i + 1, len(all_images), img_path.name, proc)
                else:
                    logger.warning(
                        "[%d/%d] %s/%s: SKIP (no face)", i + 1, len(all_images), img_path.name, proc
                    )
        logger.info("Generated %d intensity grids in %s", success, grid_dir)
        return

    logger.info("Input: %s (%d images)", input_dir, len(all_images))
    logger.info("Output: %s", output_dir)
    logger.info("Procedures: %s", procedures)
    logger.info("Intensity: %s%%", args.intensity)
    logger.info("Neural post-processing: %s", args.neural)
    if args.displacement_model:
        logger.info("Displacement model: %s", args.displacement_model)

    # Process
    all_results = []
    failed = 0
    t_start = time.time()

    for i, img_path in enumerate(all_images):
        for proc in procedures:
            result = process_image(
                img_path,
                proc,
                args.intensity,
                output_dir,
                use_neural=args.neural,
                displacement_model_path=args.displacement_model,
                save_intermediates=args.save_intermediates,
            )
            if result is None:
                failed += 1
                logger.warning(
                    "[%d/%d] %s / %s: FAILED (no face)", i + 1, len(all_images), img_path.name, proc
                )
            else:
                all_results.append(result)
                ssim = result["ssim"]
                lpips = result["lpips"]
                elapsed = result["elapsed_seconds"]
                logger.info(
                    "[%d/%d] %s / %s: SSIM=%.3f LPIPS=%.3f (%.1fs)",
                    i + 1,
                    len(all_images),
                    img_path.name,
                    proc,
                    ssim,
                    lpips,
                    elapsed,
                )

    total_time = time.time() - t_start

    # Summary report
    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_images": len(all_images),
        "total_processed": len(all_results),
        "failed": failed,
        "procedures": procedures,
        "intensity": args.intensity,
        "neural_postprocess": args.neural,
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_image": round(total_time / max(len(all_results), 1), 2),
        "results": all_results,
    }

    # Aggregate metrics
    if all_results:
        ssim_vals = [r["ssim"] for r in all_results]
        lpips_vals = [r["lpips"] for r in all_results]
        nme_vals = [r["nme"] for r in all_results]

        report["aggregate"] = {
            "ssim_mean": round(float(np.nanmean(ssim_vals)), 4),
            "ssim_std": round(float(np.nanstd(ssim_vals)), 4),
            "lpips_mean": round(float(np.nanmean(lpips_vals)), 4),
            "lpips_std": round(float(np.nanstd(lpips_vals)), 4),
            "nme_mean": round(float(np.nanmean(nme_vals)), 4),
        }

        # Per-procedure metrics
        for proc in procedures:
            proc_results = [r for r in all_results if r["procedure"] == proc]
            if proc_results:
                report["aggregate"][f"{proc}_ssim"] = round(
                    float(np.nanmean([r["ssim"] for r in proc_results])), 4
                )
                report["aggregate"][f"{proc}_lpips"] = round(
                    float(np.nanmean([r["lpips"] for r in proc_results])), 4
                )

        # Fitzpatrick breakdown
        fitz_groups: dict[str, list[dict]] = {}
        for r in all_results:
            fitz_groups.setdefault(r["fitzpatrick"], []).append(r)
        report["fitzpatrick_breakdown"] = {}
        for ftype, group in sorted(fitz_groups.items()):
            report["fitzpatrick_breakdown"][ftype] = {
                "count": len(group),
                "ssim_mean": round(float(np.nanmean([r["ssim"] for r in group])), 4),
                "lpips_mean": round(float(np.nanmean([r["lpips"] for r in group])), 4),
            }

    # Save report
    report_path = output_dir / "batch_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    logger.info("=" * 60)
    logger.info("Batch Inference Complete")
    logger.info("=" * 60)
    logger.info("Processed: %d / %d images", len(all_results), len(all_images))
    logger.info("Failed: %d", failed)
    logger.info(
        "Total time: %.1fs (%.1fs/image)", total_time, total_time / max(len(all_results), 1)
    )
    if "aggregate" in report:
        agg = report["aggregate"]
        logger.info("SSIM: %.4f +/- %.4f", agg["ssim_mean"], agg["ssim_std"])
        logger.info("LPIPS: %.4f +/- %.4f", agg["lpips_mean"], agg["lpips_std"])
        logger.info("NME: %.4f", agg["nme_mean"])
    if "fitzpatrick_breakdown" in report:
        logger.info("Fitzpatrick breakdown:")
        for ftype, data in sorted(report["fitzpatrick_breakdown"].items()):
            logger.info(
                "  Type %s: n=%d, SSIM=%.4f, LPIPS=%.4f",
                ftype,
                data["count"],
                data["ssim_mean"],
                data["lpips_mean"],
            )
    logger.info("Report saved: %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
