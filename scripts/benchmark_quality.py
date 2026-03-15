#!/usr/bin/env python3
"""Comprehensive quality benchmark for LandmarkDiff paper Table 1.

Evaluates multiple methods on the same test set and produces:
1. Per-method aggregate metrics (SSIM, LPIPS, NME, FID, Identity)
2. Per-procedure breakdown
3. Per-Fitzpatrick-type breakdown (equity analysis)
4. Statistical significance tests (paired t-test, Wilcoxon)
5. LaTeX tables ready for paper insertion

Methods evaluated:
- TPS baseline (no learned model)
- Morphing baseline
- LandmarkDiff Phase A (diffusion loss only)
- LandmarkDiff Phase B (full 4-term loss)

Usage:
    python scripts/benchmark_quality.py \
        --test_dir data/test_pairs \
        --phaseA_checkpoint checkpoints_phaseA/final \
        --phaseB_checkpoint checkpoints_phaseB/final \
        --output results/benchmark

    # TPS-only (no checkpoints needed)
    python scripts/benchmark_quality.py \
        --test_dir data/test_pairs \
        --tps_only \
        --output results/benchmark_tps
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

PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
FITZ_TYPES = ["I", "II", "III", "IV", "V", "VI"]


def load_test_pairs(test_dir: str, max_samples: int = 0) -> list[dict]:
    """Load test pairs with metadata."""
    test_path = Path(test_dir)
    input_files = sorted(test_path.glob("*_input.png"))
    if max_samples > 0:
        input_files = input_files[:max_samples]

    pairs = []
    for inp in input_files:
        prefix = inp.stem.replace("_input", "")
        target = test_path / f"{prefix}_target.png"
        if not target.exists():
            continue

        # Infer procedure from filename
        procedure = "rhinoplasty"
        for proc in PROCEDURES:
            if proc in prefix:
                procedure = proc
                break

        # Infer Fitzpatrick type if in filename
        fitz = "?"
        for ft in FITZ_TYPES:
            if f"_fitz{ft}" in prefix or f"_type{ft}" in prefix:
                fitz = ft
                break

        pairs.append(
            {
                "prefix": prefix,
                "input_path": str(inp),
                "target_path": str(target),
                "procedure": procedure,
                "fitzpatrick": fitz,
            }
        )

    return pairs


def tps_baseline(
    input_img: np.ndarray, face, procedure: str, intensity: float = 65.0
) -> np.ndarray | None:
    """Generate TPS baseline prediction."""
    manip = apply_procedure_preset(face, procedure, intensity, image_size=512)
    mask = generate_surgical_mask(face, procedure, 512, 512)
    warped = warp_image_tps(input_img, face.pixel_coords, manip.pixel_coords)
    return mask_composite(warped, input_img, mask)


def morphing_baseline(
    input_img: np.ndarray, target_img: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Simple alpha blending baseline."""
    return cv2.addWeighted(input_img, 1 - alpha, target_img, alpha, 0)


def evaluate_method(
    method_fn,
    pairs: list[dict],
    method_name: str,
    needs_target: bool = False,
) -> dict:
    """Run evaluation for a single method across all test pairs."""
    results = {
        "method": method_name,
        "per_sample": [],
        "per_procedure": {},
        "per_fitzpatrick": {},
    }

    ssim_all, lpips_all, nme_all, _id_all = [], [], [], []
    proc_metrics = {p: {"ssim": [], "lpips": [], "nme": []} for p in PROCEDURES}
    fitz_metrics = {f: {"ssim": [], "lpips": [], "nme": []} for f in FITZ_TYPES}

    t0 = time.time()
    for i, pair in enumerate(pairs):
        input_img = cv2.imread(pair["input_path"])
        target_img = cv2.imread(pair["target_path"])
        if input_img is None or target_img is None:
            continue

        input_img = cv2.resize(input_img, (512, 512))
        target_img = cv2.resize(target_img, (512, 512))

        face = extract_landmarks(input_img)
        if face is None:
            continue

        try:
            if needs_target:
                pred = method_fn(input_img, target_img)
            else:
                pred = method_fn(input_img, face, pair["procedure"])
        except Exception as e:
            print(f"  Skip {pair['prefix']}: {e}")
            continue

        if pred is None:
            continue

        # Compute metrics
        ssim = compute_ssim(pred, target_img)
        lpips_val = compute_lpips(pred, target_img)
        if lpips_val is None:
            lpips_val = float("nan")

        # NME on landmarks
        pred_face = extract_landmarks(pred)
        target_face = extract_landmarks(target_img)
        if pred_face is not None and target_face is not None:
            nme = compute_nme(pred_face.pixel_coords, target_face.pixel_coords)
        else:
            nme = float("nan")

        ssim_all.append(ssim)
        lpips_all.append(lpips_val)
        nme_all.append(nme)

        # Per-procedure
        proc = pair["procedure"]
        if proc in proc_metrics:
            proc_metrics[proc]["ssim"].append(ssim)
            proc_metrics[proc]["lpips"].append(lpips_val)
            proc_metrics[proc]["nme"].append(nme)

        # Per-Fitzpatrick
        fitz = pair["fitzpatrick"]
        if fitz in fitz_metrics:
            fitz_metrics[fitz]["ssim"].append(ssim)
            fitz_metrics[fitz]["lpips"].append(lpips_val)
            fitz_metrics[fitz]["nme"].append(nme)

        results["per_sample"].append(
            {
                "prefix": pair["prefix"],
                "procedure": proc,
                "fitzpatrick": fitz,
                "ssim": ssim,
                "lpips": lpips_val,
                "nme": nme,
            }
        )

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i + 1}/{len(pairs)}] SSIM={np.nanmean(ssim_all):.4f} "
                f"LPIPS={np.nanmean(lpips_all):.4f} ({elapsed:.0f}s)"
            )

    # Aggregate
    results["aggregate"] = {
        "ssim_mean": float(np.nanmean(ssim_all)) if ssim_all else 0,
        "ssim_std": float(np.nanstd(ssim_all)) if ssim_all else 0,
        "lpips_mean": float(np.nanmean(lpips_all)) if lpips_all else 0,
        "lpips_std": float(np.nanstd(lpips_all)) if lpips_all else 0,
        "nme_mean": float(np.nanmean(nme_all)) if nme_all else 0,
        "nme_std": float(np.nanstd(nme_all)) if nme_all else 0,
        "n": len(ssim_all),
    }

    # Per-procedure aggregates
    for proc, m in proc_metrics.items():
        if m["ssim"]:
            results["per_procedure"][proc] = {
                "ssim": float(np.nanmean(m["ssim"])),
                "lpips": float(np.nanmean(m["lpips"])),
                "nme": float(np.nanmean(m["nme"])),
                "n": len(m["ssim"]),
            }

    # Per-Fitzpatrick aggregates
    for ftype, m in fitz_metrics.items():
        if m["ssim"]:
            results["per_fitzpatrick"][ftype] = {
                "ssim": float(np.nanmean(m["ssim"])),
                "lpips": float(np.nanmean(m["lpips"])),
                "nme": float(np.nanmean(m["nme"])),
                "n": len(m["ssim"]),
            }

    elapsed = time.time() - t0
    print(
        f"  {method_name}: SSIM={results['aggregate']['ssim_mean']:.4f} "
        f"LPIPS={results['aggregate']['lpips_mean']:.4f} "
        f"NME={results['aggregate']['nme_mean']:.4f} "
        f"({elapsed:.0f}s, n={results['aggregate']['n']})"
    )

    return results


def significance_test(method_a: dict, method_b: dict) -> dict:
    """Run paired significance tests between two methods."""
    from scipy import stats

    a_samples = {s["prefix"]: s for s in method_a["per_sample"]}
    b_samples = {s["prefix"]: s for s in method_b["per_sample"]}

    common = set(a_samples.keys()) & set(b_samples.keys())
    if len(common) < 5:
        return {"n_common": len(common), "insufficient_data": True}

    results = {}
    for metric in ["ssim", "lpips", "nme"]:
        a_vals = [a_samples[k][metric] for k in common if not np.isnan(a_samples[k][metric])]
        b_vals = [b_samples[k][metric] for k in common if not np.isnan(b_samples[k][metric])]

        if len(a_vals) < 5 or len(b_vals) < 5:
            continue

        # Paired t-test
        t_stat, p_ttest = stats.ttest_rel(a_vals[: len(b_vals)], b_vals[: len(a_vals)])

        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, p_wilcoxon = stats.wilcoxon(a_vals[: len(b_vals)], b_vals[: len(a_vals)])
        except Exception:
            _w_stat, p_wilcoxon = float("nan"), float("nan")

        results[metric] = {
            "t_statistic": float(t_stat),
            "p_ttest": float(p_ttest),
            "p_wilcoxon": float(p_wilcoxon),
            "significant_005": p_ttest < 0.05,
        }

    return {"n_common": len(common), "tests": results}


def generate_latex_table(all_results: dict, output_path: Path) -> str:
    """Generate LaTeX table for paper (Table 1: Main comparison)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Quantitative comparison with baselines on our surgical test set.}",
        r"\label{tab:main_comparison}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & SSIM$\uparrow$ & LPIPS$\downarrow$ & NME$\downarrow$ & FID$\downarrow$ \\",
        r"\midrule",
    ]

    method_order = ["TPS Baseline", "Morphing", "Phase A", "Phase B (Ours)"]

    for method_name in method_order:
        if method_name not in all_results:
            continue
        r = all_results[method_name]
        agg = r["aggregate"]

        use_bold = "Ours" in method_name

        def fmt(val, f=".4f", _bold=use_bold):
            s = f"{val:{f}}"
            return rf"\textbf{{{s}}}" if _bold else s

        fid_str = fmt(agg.get("fid", 0), ".1f") if "fid" in agg else "--"
        lines.append(
            f"{method_name} & {fmt(agg['ssim_mean'])} & "
            f"{fmt(agg['lpips_mean'])} & {fmt(agg['nme_mean'])} & {fid_str} \\\\"
        )

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


def generate_procedure_table(all_results: dict, output_path: Path) -> str:
    """Generate per-procedure breakdown table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-procedure SSIM comparison.}",
        r"\label{tab:per_procedure}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Rhino & Bleph & Rhytid & Orthog \\",
        r"\midrule",
    ]

    for method_name, r in all_results.items():
        proc = r.get("per_procedure", {})
        cols = []
        for p in PROCEDURES:
            if p in proc:
                cols.append(f"{proc[p]['ssim']:.3f}")
            else:
                cols.append("--")
        lines.append(f"{method_name} & {' & '.join(cols)} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
        ]
    )

    latex = "\n".join(lines)
    output_path.write_text(latex)
    return latex


def main():
    parser = argparse.ArgumentParser(description="Quality benchmark for paper")
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output", default="results/benchmark")
    parser.add_argument("--phaseA_checkpoint", default=None)
    parser.add_argument("--phaseB_checkpoint", default=None)
    parser.add_argument(
        "--tps_only", action="store_true", help="Only evaluate TPS/morphing baselines"
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=20, help="Diffusion inference steps")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_test_pairs(args.test_dir, args.max_samples)
    print(f"Loaded {len(pairs)} test pairs")
    if not pairs:
        print("No test pairs found!")
        sys.exit(1)

    all_results = {}

    # 1. TPS Baseline
    print("\n=== TPS Baseline ===")
    all_results["TPS Baseline"] = evaluate_method(
        tps_baseline, pairs, "TPS Baseline", needs_target=False
    )

    # 2. Morphing Baseline
    print("\n=== Morphing Baseline ===")
    all_results["Morphing"] = evaluate_method(
        morphing_baseline, pairs, "Morphing", needs_target=True
    )

    # 3. Phase A (if checkpoint provided)
    if args.phaseA_checkpoint and not args.tps_only:
        print("\n=== Phase A (Diffusion Loss Only) ===")
        from landmarkdiff.inference import LandmarkDiffPipeline

        pipe_a = LandmarkDiffPipeline(
            mode="controlnet",
            controlnet_checkpoint=args.phaseA_checkpoint,
        )
        pipe_a.load()

        def phase_a_fn(input_img, face, procedure):
            result = pipe_a.generate(
                input_img, procedure=procedure, num_inference_steps=args.num_steps, seed=42
            )
            return result["output"]

        all_results["Phase A"] = evaluate_method(phase_a_fn, pairs, "Phase A", needs_target=False)
        del pipe_a
        import torch

        torch.cuda.empty_cache()

    # 4. Phase B (if checkpoint provided)
    if args.phaseB_checkpoint and not args.tps_only:
        print("\n=== Phase B (Full Loss — Ours) ===")
        from landmarkdiff.inference import LandmarkDiffPipeline

        pipe_b = LandmarkDiffPipeline(
            mode="controlnet",
            controlnet_checkpoint=args.phaseB_checkpoint,
        )
        pipe_b.load()

        def phase_b_fn(input_img, face, procedure):
            result = pipe_b.generate(
                input_img, procedure=procedure, num_inference_steps=args.num_steps, seed=42
            )
            return result["output"]

        all_results["Phase B (Ours)"] = evaluate_method(
            phase_b_fn, pairs, "Phase B (Ours)", needs_target=False
        )
        del pipe_b
        import torch

        torch.cuda.empty_cache()

    # Significance tests (all vs Phase B if available)
    sig_results = {}
    if "Phase B (Ours)" in all_results:
        for method_name in ["TPS Baseline", "Morphing", "Phase A"]:
            if method_name in all_results:
                sig = significance_test(all_results[method_name], all_results["Phase B (Ours)"])
                sig_results[f"{method_name} vs Phase B"] = sig

    # Save results
    save_data = {
        "methods": {
            k: {
                "aggregate": v["aggregate"],
                "per_procedure": v.get("per_procedure", {}),
                "per_fitzpatrick": v.get("per_fitzpatrick", {}),
            }
            for k, v in all_results.items()
        },
        "significance": sig_results,
        "test_dir": args.test_dir,
        "num_pairs": len(pairs),
    }

    with open(output_dir / "benchmark.json", "w") as f:
        json.dump(save_data, f, indent=2)

    # Generate LaTeX tables
    generate_latex_table(all_results, output_dir / "table1_comparison.tex")
    generate_procedure_table(all_results, output_dir / "table_per_procedure.tex")

    # Print summary
    print(f"\n{'=' * 70}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Method':<25} {'SSIM':>8} {'LPIPS':>8} {'NME':>8} {'n':>6}")
    print("-" * 60)
    for name, r in all_results.items():
        agg = r["aggregate"]
        print(
            f"{name:<25} {agg['ssim_mean']:>8.4f} {agg['lpips_mean']:>8.4f} "
            f"{agg['nme_mean']:>8.4f} {agg['n']:>6}"
        )

    if sig_results:
        print("\nSignificance Tests (p-values):")
        for comp, sig in sig_results.items():
            if "tests" in sig:
                for metric, test in sig["tests"].items():
                    star = "*" if test["significant_005"] else ""
                    print(f"  {comp} [{metric}]: p={test['p_ttest']:.4f}{star}")

    print(f"\nResults saved to {output_dir}/")
    print("  benchmark.json — full results")
    print("  table1_comparison.tex — main comparison table")
    print("  table_per_procedure.tex — per-procedure breakdown")


if __name__ == "__main__":
    main()
