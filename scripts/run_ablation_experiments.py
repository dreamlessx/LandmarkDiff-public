#!/usr/bin/env python3
"""Run ablation experiments for Table 3 of the paper.

Generates ablation results by evaluating different conditioning
configurations using TPS baselines (no training needed) and optionally
trained checkpoints.

Ablation conditions (from Table 3 spec):
1. Mesh wireframe only
2. Canny edges only
3. Mesh + Canny (our conditioning)
4. No surgical mask
5. No EMA
6. No curriculum
7. Phase A only (no Phase B fine-tuning)
8. Full pipeline (Ours)

For TPS-based ablations (mask, conditioning), this can run on CPU.
For training-based ablations (EMA, curriculum, Phase A/B), requires
pre-trained checkpoints.

Usage:
    # TPS ablations only (no GPU needed)
    python scripts/run_ablation_experiments.py \
        --test_dir data/splits/test --mode tps --max_samples 50

    # Full ablation with checkpoints
    python scripts/run_ablation_experiments.py \
        --test_dir data/splits/test --mode full \
        --checkpoint_full checkpoints_phaseB/final \
        --checkpoint_phaseA checkpoints_phaseA/final \
        --checkpoint_no_ema checkpoints_no_ema/final \
        --checkpoint_no_curriculum checkpoints_no_curriculum/final
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.evaluation import compute_lpips, compute_nme, compute_ssim
from landmarkdiff.inference import mask_composite
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps

PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]


def load_test_images(test_dir: str, max_samples: int = 0) -> list[dict]:
    """Load test images with metadata."""
    test_path = Path(test_dir)
    input_files = sorted(test_path.glob("*_input.png"))
    if max_samples > 0:
        input_files = input_files[:max_samples]

    images = []
    for inp in input_files:
        prefix = inp.stem.replace("_input", "")
        target = test_path / f"{prefix}_target.png"
        if not target.exists():
            continue

        procedure = "rhinoplasty"
        for proc in PROCEDURES:
            if proc in prefix:
                procedure = proc
                break

        images.append(
            {
                "prefix": prefix,
                "input_path": str(inp),
                "target_path": str(target),
                "procedure": procedure,
            }
        )

    return images


def ablation_tps_full(input_img, face, procedure, image_size=512):
    """Full TPS pipeline: mesh+canny conditioning, mask, composite."""
    manip = apply_procedure_preset(face, procedure, 65.0, image_size=image_size)
    mask = generate_surgical_mask(face, procedure, image_size, image_size)
    warped = warp_image_tps(input_img, face.pixel_coords, manip.pixel_coords)
    return mask_composite(warped, input_img, mask)


def ablation_tps_no_mask(input_img, face, procedure, image_size=512):
    """TPS without surgical mask compositing (warped directly)."""
    manip = apply_procedure_preset(face, procedure, 65.0, image_size=image_size)
    warped = warp_image_tps(input_img, face.pixel_coords, manip.pixel_coords)
    return warped


def ablation_tps_half_intensity(input_img, face, procedure, image_size=512):
    """TPS with half intensity (50% displacement)."""
    manip = apply_procedure_preset(face, procedure, 32.5, image_size=image_size)
    mask = generate_surgical_mask(face, procedure, image_size, image_size)
    warped = warp_image_tps(input_img, face.pixel_coords, manip.pixel_coords)
    return mask_composite(warped, input_img, mask)


def ablation_tps_double_intensity(input_img, face, procedure, image_size=512):
    """TPS with double intensity (130% displacement)."""
    manip = apply_procedure_preset(face, procedure, 130.0, image_size=image_size)
    mask = generate_surgical_mask(face, procedure, image_size, image_size)
    warped = warp_image_tps(input_img, face.pixel_coords, manip.pixel_coords)
    return mask_composite(warped, input_img, mask)


def evaluate_ablation(
    method_fn,
    test_images: list[dict],
    ablation_name: str,
) -> dict:
    """Evaluate a single ablation configuration."""
    ssim_all, lpips_all, nme_all = [], [], []
    proc_metrics = {p: {"ssim": [], "lpips": [], "nme": []} for p in PROCEDURES}

    t0 = time.time()
    skipped = 0

    for i, item in enumerate(test_images):
        input_img = cv2.imread(item["input_path"])
        target_img = cv2.imread(item["target_path"])
        if input_img is None or target_img is None:
            skipped += 1
            continue

        input_img = cv2.resize(input_img, (512, 512))
        target_img = cv2.resize(target_img, (512, 512))

        face = extract_landmarks(input_img)
        if face is None:
            skipped += 1
            continue

        try:
            pred = method_fn(input_img, face, item["procedure"])
        except Exception as e:
            print(f"  Skip {item['prefix']}: {e}")
            skipped += 1
            continue

        if pred is None:
            skipped += 1
            continue

        ssim = compute_ssim(pred, target_img)
        lpips_val = compute_lpips(pred, target_img)
        if lpips_val is None:
            lpips_val = float("nan")

        pred_face = extract_landmarks(pred)
        target_face = extract_landmarks(target_img)
        if pred_face is not None and target_face is not None:
            nme = compute_nme(pred_face.pixel_coords, target_face.pixel_coords)
        else:
            nme = float("nan")

        ssim_all.append(ssim)
        lpips_all.append(lpips_val)
        nme_all.append(nme)

        proc = item["procedure"]
        if proc in proc_metrics:
            proc_metrics[proc]["ssim"].append(ssim)
            proc_metrics[proc]["lpips"].append(lpips_val)
            proc_metrics[proc]["nme"].append(nme)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i + 1}/{len(test_images)}] SSIM={np.nanmean(ssim_all):.4f} "
                f"LPIPS={np.nanmean(lpips_all):.4f} ({elapsed:.0f}s)"
            )

    result = {
        "ablation": ablation_name,
        "n": len(ssim_all),
        "skipped": skipped,
        "aggregate": {
            "ssim_mean": float(np.nanmean(ssim_all)) if ssim_all else 0,
            "ssim_std": float(np.nanstd(ssim_all)) if ssim_all else 0,
            "lpips_mean": float(np.nanmean(lpips_all)) if lpips_all else 0,
            "lpips_std": float(np.nanstd(lpips_all)) if lpips_all else 0,
            "nme_mean": float(np.nanmean(nme_all)) if nme_all else 0,
            "nme_std": float(np.nanstd(nme_all)) if nme_all else 0,
        },
        "per_procedure": {},
    }

    for proc, m in proc_metrics.items():
        if m["ssim"]:
            result["per_procedure"][proc] = {
                "ssim": float(np.nanmean(m["ssim"])),
                "lpips": float(np.nanmean(m["lpips"])),
                "nme": float(np.nanmean(m["nme"])),
                "n": len(m["ssim"]),
            }

    elapsed = time.time() - t0
    print(
        f"  {ablation_name}: SSIM={result['aggregate']['ssim_mean']:.4f} "
        f"LPIPS={result['aggregate']['lpips_mean']:.4f} "
        f"NME={result['aggregate']['nme_mean']:.4f} ({elapsed:.0f}s, n={result['n']})"
    )

    return result


def generate_ablation_latex(results: dict, output_path: Path) -> str:
    """Generate LaTeX table for Table 3 (Ablation Study)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study. We remove one component at a time from our",
        r"full pipeline and measure quality degradation.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & SSIM$\uparrow$ & LPIPS$\downarrow$ & NME$\downarrow$ & $\Delta$SSIM \\",
        r"\midrule",
    ]

    # Find the full pipeline SSIM for delta computation
    full_ssim = None
    if "Full Pipeline (Ours)" in results:
        full_ssim = results["Full Pipeline (Ours)"]["aggregate"]["ssim_mean"]
    elif "Full TPS" in results:
        full_ssim = results["Full TPS"]["aggregate"]["ssim_mean"]

    ablation_order = [
        "Full Pipeline (Ours)",
        "Full TPS",
        "Mesh Only",
        "Canny Only",
        "Mesh + Canny",
        "No Mask",
        "Half Intensity",
        "Double Intensity",
        "No EMA",
        "No Curriculum",
        "Phase A Only",
    ]

    for name in ablation_order:
        if name not in results:
            continue
        agg = results[name]["aggregate"]
        is_ours = "Ours" in name

        ssim_s = f"{agg['ssim_mean']:.4f}"
        lpips_s = f"{agg['lpips_mean']:.4f}"
        nme_s = f"{agg['nme_mean']:.4f}"

        if full_ssim is not None:
            delta = agg["ssim_mean"] - full_ssim
            delta_s = f"{delta:+.4f}" if delta != 0 else "---"
        else:
            delta_s = "---"

        if is_ours:
            ssim_s = rf"\textbf{{{ssim_s}}}"
            lpips_s = rf"\textbf{{{lpips_s}}}"
            nme_s = rf"\textbf{{{nme_s}}}"

        lines.append(f"{name} & {ssim_s} & {lpips_s} & {nme_s} & {delta_s} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    latex = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex)
    return latex


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output", default="results/ablation")
    parser.add_argument(
        "--mode",
        choices=["tps", "full"],
        default="tps",
        help="'tps' for baseline ablations, 'full' for with checkpoints",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--checkpoint_full", default=None)
    parser.add_argument("--checkpoint_phaseA", default=None)
    parser.add_argument("--checkpoint_no_ema", default=None)
    parser.add_argument("--checkpoint_no_curriculum", default=None)
    parser.add_argument("--num_steps", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = load_test_images(args.test_dir, args.max_samples)
    print(f"Loaded {len(test_images)} test images")
    if not test_images:
        print("No test images found!")
        sys.exit(1)

    all_results = {}

    # ─── TPS-based ablations (always available) ───
    print("\n=== Full TPS (mask + composite) ===")
    all_results["Full TPS"] = evaluate_ablation(ablation_tps_full, test_images, "Full TPS")

    print("\n=== No Mask (warped only) ===")
    all_results["No Mask"] = evaluate_ablation(ablation_tps_no_mask, test_images, "No Mask")

    print("\n=== Half Intensity (50%) ===")
    all_results["Half Intensity"] = evaluate_ablation(
        ablation_tps_half_intensity, test_images, "Half Intensity"
    )

    print("\n=== Double Intensity (130%) ===")
    all_results["Double Intensity"] = evaluate_ablation(
        ablation_tps_double_intensity, test_images, "Double Intensity"
    )

    # ─── Checkpoint-based ablations (if provided) ───
    if args.mode == "full":
        import torch

        from landmarkdiff.inference import LandmarkDiffPipeline

        def _make_pipeline_fn(checkpoint_path, num_steps):
            pipe = LandmarkDiffPipeline(
                mode="controlnet",
                controlnet_checkpoint=checkpoint_path,
            )
            pipe.load()

            def fn(input_img, face, procedure):
                result = pipe.generate(
                    input_img,
                    procedure=procedure,
                    num_inference_steps=num_steps,
                    seed=42,
                )
                return result["output"]

            return fn, pipe

        if args.checkpoint_full:
            print("\n=== Full Pipeline (Ours) ===")
            fn, pipe = _make_pipeline_fn(args.checkpoint_full, args.num_steps)
            all_results["Full Pipeline (Ours)"] = evaluate_ablation(
                fn, test_images, "Full Pipeline (Ours)"
            )
            del pipe
            torch.cuda.empty_cache()

        if args.checkpoint_phaseA:
            print("\n=== Phase A Only ===")
            fn, pipe = _make_pipeline_fn(args.checkpoint_phaseA, args.num_steps)
            all_results["Phase A Only"] = evaluate_ablation(fn, test_images, "Phase A Only")
            del pipe
            torch.cuda.empty_cache()

        if args.checkpoint_no_ema:
            print("\n=== No EMA ===")
            fn, pipe = _make_pipeline_fn(args.checkpoint_no_ema, args.num_steps)
            all_results["No EMA"] = evaluate_ablation(fn, test_images, "No EMA")
            del pipe
            torch.cuda.empty_cache()

        if args.checkpoint_no_curriculum:
            print("\n=== No Curriculum ===")
            fn, pipe = _make_pipeline_fn(args.checkpoint_no_curriculum, args.num_steps)
            all_results["No Curriculum"] = evaluate_ablation(fn, test_images, "No Curriculum")
            del pipe
            torch.cuda.empty_cache()

    # ─── Save results ───
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate LaTeX table
    generate_ablation_latex(all_results, output_dir / "table3_ablation.tex")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("ABLATION RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Configuration':<30} {'SSIM':>8} {'LPIPS':>8} {'NME':>8} {'n':>5}")
    print("-" * 65)
    for name, r in all_results.items():
        agg = r["aggregate"]
        print(
            f"{name:<30} {agg['ssim_mean']:>8.4f} {agg['lpips_mean']:>8.4f} "
            f"{agg['nme_mean']:>8.4f} {r['n']:>5}"
        )

    print(f"\nResults saved to {output_dir}/")
    print("  ablation_results.json")
    print("  table3_ablation.tex")


if __name__ == "__main__":
    main()
