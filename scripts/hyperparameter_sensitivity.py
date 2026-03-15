"""Hyperparameter sensitivity analysis for LandmarkDiff.

Analyzes how changes in key hyperparameters affect output quality.
Useful for finding optimal settings and understanding model behavior.

Usage:
    python scripts/hyperparameter_sensitivity.py IMAGE --procedure rhinoplasty
    python scripts/hyperparameter_sensitivity.py IMAGE --sweep intensity
    python scripts/hyperparameter_sensitivity.py IMAGE --sweep guidance_scale
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

SWEEP_CONFIGS = {
    "intensity": {
        "description": "Surgical intensity (displacement magnitude)",
        "values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "param": "intensity",
        "default": 65,
    },
    "guidance_scale": {
        "description": "Classifier-free guidance scale",
        "values": [3.0, 5.0, 7.0, 7.5, 9.0, 10.0, 12.0, 15.0],
        "param": "guidance_scale",
        "default": 7.5,
    },
    "controlnet_scale": {
        "description": "ControlNet conditioning scale",
        "values": [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5],
        "param": "controlnet_conditioning_scale",
        "default": 1.0,
    },
    "inference_steps": {
        "description": "Number of denoising steps",
        "values": [10, 15, 20, 25, 30, 40, 50],
        "param": "num_inference_steps",
        "default": 30,
    },
    "codeformer_fidelity": {
        "description": "CodeFormer fidelity weight",
        "values": [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "param": "codeformer_fidelity",
        "default": 0.7,
    },
}


# ---------------------------------------------------------------------------
# TPS-based intensity sweep (no GPU required)
# ---------------------------------------------------------------------------


def sweep_intensity_tps(
    image: np.ndarray,
    procedure: str,
    values: list[float] | None = None,
    output_dir: str = "sensitivity_output",
) -> dict:
    """Sweep surgical intensity using TPS-based pipeline (no GPU needed).

    Args:
        image: Input BGR image.
        procedure: Surgical procedure type.
        values: Intensity values to sweep.
        output_dir: Output directory for results.

    Returns:
        Dict with sweep results and metrics.
    """
    from landmarkdiff.landmarks import extract_landmarks, render_landmark_image
    from landmarkdiff.manipulation import apply_procedure_preset
    from landmarkdiff.masking import generate_surgical_mask
    from landmarkdiff.synthetic.tps_warp import warp_image_tps

    if values is None:
        values = SWEEP_CONFIGS["intensity"]["values"]

    out_dir = Path(output_dir) / f"intensity_sweep_{procedure}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract landmarks once
    face = extract_landmarks(image)
    if face is None:
        print("ERROR: No face detected in image")
        return {"error": "no_face"}

    results = []
    h, w = image.shape[:2]

    for intensity in values:
        t0 = time.perf_counter()

        # Apply displacement
        manip = apply_procedure_preset(face, procedure, intensity, image_size=max(h, w))

        # Compute displacement magnitude
        displacement = np.abs(manip.landmarks - face.landmarks).mean()

        # Apply TPS warp
        src = face.pixel_coords
        dst = manip.pixel_coords

        try:
            warped = warp_image_tps(image, src, dst)
        except Exception:
            warped = image.copy()

        # Generate mask and conditioning
        mask = generate_surgical_mask(face, procedure, w, h)
        cond = render_landmark_image(manip, w, h)

        # Compute simple quality metrics (no model needed)
        # SSIM approximation via structural correlation
        diff = np.abs(warped.astype(float) - image.astype(float))
        mse = diff.mean()
        psnr = 10 * np.log10(255**2 / max(mse, 1e-10))

        # Mask region change
        mask_region = mask > 0.5
        if mask_region.sum() > 0:
            region_diff = diff[mask_region].mean()
        else:
            region_diff = 0

        elapsed = time.perf_counter() - t0

        result = {
            "intensity": intensity,
            "displacement_mean": round(float(displacement), 6),
            "mse": round(float(mse), 2),
            "psnr": round(float(psnr), 2),
            "mask_region_diff": round(float(region_diff), 2),
            "elapsed_ms": round(elapsed * 1000, 1),
        }
        results.append(result)

        # Save outputs
        cv2.imwrite(str(out_dir / f"intensity_{intensity:03.0f}_warped.png"), warped)
        cv2.imwrite(str(out_dir / f"intensity_{intensity:03.0f}_cond.png"), cond)

        print(
            f"  intensity={intensity:5.1f}: displacement={displacement:.6f}, "
            f"MSE={mse:.1f}, PSNR={psnr:.1f}dB, time={elapsed * 1000:.0f}ms"
        )

    # Save composite comparison
    _save_comparison_grid(
        out_dir / "comparison_grid.png",
        [cv2.imread(str(out_dir / f"intensity_{v:03.0f}_warped.png")) for v in values],
        [f"I={v}" for v in values],
        image,
    )

    # Save results
    import json

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return {"results": results, "output_dir": str(out_dir)}


def _save_comparison_grid(
    output_path: Path,
    images: list[np.ndarray],
    labels: list[str],
    original: np.ndarray,
    cols: int = 5,
) -> None:
    """Save a grid of comparison images."""
    # Add original as first image
    all_images = [original] + images
    all_labels = ["Original"] + labels

    n = len(all_images)
    rows = (n + cols - 1) // cols

    # Resize all to same size
    h, w = 256, 256
    resized = []
    for img in all_images:
        if img is not None:
            resized.append(cv2.resize(img, (w, h)))
        else:
            resized.append(np.zeros((h, w, 3), dtype=np.uint8))

    # Create grid
    grid_h = rows * (h + 30)  # Extra space for labels
    grid_w = cols * w
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    for i, (img, label) in enumerate(zip(resized, all_labels)):
        row = i // cols
        col = i % cols
        y = row * (h + 30)
        x = col * w

        grid[y : y + h, x : x + w] = img

        # Add label
        cv2.putText(grid, label, (x + 5, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imwrite(str(output_path), grid)


# ---------------------------------------------------------------------------
# Sensitivity report
# ---------------------------------------------------------------------------


def generate_sensitivity_report(
    sweep_results: dict,
    param_name: str,
) -> str:
    """Generate a text report from sweep results."""
    results = sweep_results.get("results", [])
    if not results:
        return "No results to report."

    lines = [
        "=" * 60,
        f"Hyperparameter Sensitivity: {param_name}",
        "=" * 60,
        "",
    ]

    # Table header
    header = f"{'Value':>8} {'Displacement':>14} {'MSE':>8} {'PSNR':>8} {'Region Diff':>12} {'Time (ms)':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        lines.append(
            f"{r.get('intensity', r.get('value', '')):>8.1f} "
            f"{r['displacement_mean']:>14.6f} "
            f"{r['mse']:>8.1f} "
            f"{r['psnr']:>8.1f} "
            f"{r['mask_region_diff']:>12.1f} "
            f"{r['elapsed_ms']:>10.0f}"
        )

    # Summary
    disp_values = [r["displacement_mean"] for r in results]
    mse_values = [r["mse"] for r in results]
    lines.append("")
    lines.append(f"Displacement range: [{min(disp_values):.6f}, {max(disp_values):.6f}]")
    lines.append(f"MSE range: [{min(mse_values):.1f}, {max(mse_values):.1f}]")

    # Find sweet spot (good displacement with reasonable MSE)
    if len(results) > 2:
        # Normalize both metrics
        d_norm = (np.array(disp_values) - min(disp_values)) / max(
            max(disp_values) - min(disp_values), 1e-10
        )
        m_norm = (np.array(mse_values) - min(mse_values)) / max(
            max(mse_values) - min(mse_values), 1e-10
        )
        # Score: high displacement, low MSE
        scores = d_norm - 0.5 * m_norm
        best_idx = int(np.argmax(scores))
        best_val = results[best_idx].get("intensity", results[best_idx].get("value", "?"))
        lines.append(f"\nRecommended value: {best_val} (best displacement/quality trade-off)")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity analysis")
    parser.add_argument("image", nargs="?", help="Input face image")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic image")
    parser.add_argument(
        "--procedure",
        default="rhinoplasty",
        choices=["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"],
    )
    parser.add_argument(
        "--sweep",
        default="intensity",
        choices=list(SWEEP_CONFIGS.keys()),
        help="Parameter to sweep",
    )
    parser.add_argument("--output", default="sensitivity_output", help="Output directory")
    args = parser.parse_args()

    # Load image
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"ERROR: Cannot read {args.image}")
            sys.exit(1)
        image = cv2.resize(image, (512, 512))
    elif args.synthetic:
        from scripts.verify_pipeline import create_synthetic_test_image

        image = create_synthetic_test_image()
        print("Using synthetic test image")
    else:
        # Try to find a real image
        data_dir = Path(__file__).resolve().parent.parent / "data"
        test_images = list(data_dir.glob("**/celeba_hq_extracted/*.png"))[:1]
        if test_images:
            image = cv2.imread(str(test_images[0]))
            image = cv2.resize(image, (512, 512))
            print(f"Using: {test_images[0].name}")
        else:
            from scripts.verify_pipeline import create_synthetic_test_image

            image = create_synthetic_test_image()
            print("No test images found, using synthetic")

    print(f"\nSweeping {args.sweep} for {args.procedure}")
    print("-" * 40)

    if args.sweep == "intensity":
        result = sweep_intensity_tps(image, args.procedure, output_dir=args.output)
    else:
        sweep_cfg = SWEEP_CONFIGS[args.sweep]
        print(f"Note: Sweeping {args.sweep} requires model inference.")
        print(f"Values to try: {sweep_cfg['values']}")
        print(f"Default: {sweep_cfg['default']}")
        print("\nFor TPS-only mode, use --sweep intensity")
        result = {"results": []}

    if result.get("results"):
        report = generate_sensitivity_report(result, args.sweep)
        print()
        print(report)

        # Save report
        report_path = Path(args.output) / "sensitivity_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
