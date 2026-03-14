"""Multi-seed evaluation for robust metric reporting.

Runs evaluate_checkpoint.py with multiple random seeds and aggregates
results as mean ± std, which is what Table 1 in the paper reports.

Usage:
    python scripts/multi_seed_eval.py \
        --checkpoint checkpoints/phaseB/best \
        --test-dir data/hda_splits/test \
        --output paper/eval_results_multiseed.json \
        --seeds 42 123 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_single_seed(
    checkpoint: str,
    test_dir: str,
    seed: int,
    num_steps: int = 20,
    save_images: bool = False,
    images_dir: str | None = None,
) -> dict:
    """Run evaluation with a single seed, returning the results dict."""
    from scripts.evaluate_checkpoint import evaluate

    output_path = str(ROOT / "paper" / f"eval_results_seed{seed}.json")
    return evaluate(
        checkpoint_path=checkpoint,
        test_dir=test_dir,
        output_path=output_path,
        num_steps=num_steps,
        seed=seed,
        save_images=save_images,
        images_dir=images_dir,
    )


def aggregate_seeds(all_results: list[dict], seeds: list[int]) -> dict:
    """Aggregate results across seeds into mean ± std format.

    The output JSON has the same structure as a single-seed result but
    with each scalar replaced by {"mean": ..., "std": ..., "per_seed": [...]}.
    """
    n_seeds = len(all_results)
    agg = {
        "n_seeds": n_seeds,
        "seeds": seeds,
        "num_pairs": all_results[0].get("num_pairs", 0),
        "checkpoint": all_results[0].get("checkpoint", ""),
    }

    # ── Overall metrics ─────────────────────────────────────────────────
    metric_keys = ["fid", "ssim", "lpips", "nme", "identity_sim"]
    agg["metrics"] = {}
    for key in metric_keys:
        vals = [r["metrics"].get(key, 0.0) for r in all_results]
        agg["metrics"][key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "per_seed": vals,
        }

    # ── Per-procedure ────────────────────────────────────────────────────
    all_procs = set()
    for r in all_results:
        all_procs.update(r.get("per_procedure", {}).keys())

    agg["per_procedure"] = {}
    for proc in sorted(all_procs):
        agg["per_procedure"][proc] = {}
        for key in metric_keys:
            vals = []
            for r in all_results:
                pp = r.get("per_procedure", {}).get(proc, {})
                vals.append(
                    pp.get(key, pp.get("identity", pp.get("identity_sim", 0.0)))
                    if key == "identity_sim"
                    else pp.get(key, 0.0)
                )
            agg["per_procedure"][proc][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "per_seed": vals,
            }
        # Count is constant across seeds
        counts = [r.get("per_procedure", {}).get(proc, {}).get("count", 0) for r in all_results]
        agg["per_procedure"][proc]["n"] = max(counts)

    # ── Per-Fitzpatrick ──────────────────────────────────────────────────
    all_fitz = set()
    for r in all_results:
        all_fitz.update(r.get("per_fitzpatrick", {}).keys())

    agg["per_fitzpatrick"] = {}
    for fitz in sorted(all_fitz):
        agg["per_fitzpatrick"][fitz] = {}
        for key in metric_keys:
            vals = []
            for r in all_results:
                pf = r.get("per_fitzpatrick", {}).get(fitz, {})
                vals.append(
                    pf.get(key, pf.get("identity", pf.get("identity_sim", 0.0)))
                    if key == "identity_sim"
                    else pf.get(key, 0.0)
                )
            agg["per_fitzpatrick"][fitz][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "per_seed": vals,
            }
        counts = [r.get("per_fitzpatrick", {}).get(fitz, {}).get("count", 0) for r in all_results]
        agg["per_fitzpatrick"][fitz]["n"] = max(counts)

    return agg


def print_summary(agg: dict) -> None:
    """Print a nicely formatted summary table."""
    seeds = agg["seeds"]
    print(f"\n{'=' * 72}")
    print(f"MULTI-SEED EVALUATION  (seeds: {seeds})")
    print(f"{'=' * 72}")

    # Overall
    m = agg["metrics"]
    print(f"\nOverall ({agg['num_pairs']} pairs):")
    print(f"  {'Metric':<15s} {'Mean':>8s} {'± Std':>8s}  per-seed")
    print(f"  {'-' * 60}")
    for key in ["fid", "ssim", "lpips", "nme", "identity_sim"]:
        v = m[key]
        per = ", ".join(f"{x:.4f}" for x in v["per_seed"])
        print(f"  {key:<15s} {v['mean']:>8.4f} {v['std']:>8.4f}  [{per}]")

    # Per procedure
    print("\nPer Procedure:")
    for proc in sorted(agg["per_procedure"]):
        pp = agg["per_procedure"][proc]
        n = pp.get("n", "?")
        print(f"\n  {proc} (n={n}):")
        for key in ["ssim", "lpips", "nme", "identity_sim"]:
            v = pp[key]
            print(f"    {key:<15s} {v['mean']:>8.4f} ± {v['std']:.4f}")

    # Per Fitzpatrick
    if agg.get("per_fitzpatrick"):
        print("\nPer Fitzpatrick Type:")
        for fitz in sorted(agg["per_fitzpatrick"]):
            pf = agg["per_fitzpatrick"][fitz]
            n = pf.get("n", "?")
            print(f"\n  {fitz} (n={n}):")
            for key in ["ssim", "lpips", "nme", "identity_sim"]:
                v = pf[key]
                print(f"    {key:<15s} {v['mean']:>8.4f} ± {v['std']:.4f}")


def generate_latex_rows(agg: dict) -> str:
    """Generate LaTeX table rows in mean±std format for Table 1."""
    lines = []
    for proc in sorted(agg["per_procedure"]):
        pp = agg["per_procedure"][proc]
        fid = pp.get("fid", {})
        lpips = pp.get("lpips", {})
        nme = pp.get("nme", {})
        ssim = pp.get("ssim", {})
        arc = pp.get("identity_sim", {})

        fid_str = f"{fid['mean']:.1f}" if fid.get("mean", 0) > 0 else "--"
        lp_str = f"{lpips['mean']:.3f}$\\pm${lpips['std']:.3f}"
        nme_str = f"{nme['mean']:.3f}$\\pm${nme['std']:.3f}"
        ss_str = f"{ssim['mean']:.3f}$\\pm${ssim['std']:.3f}"
        arc_str = f"{arc['mean']:.3f}$\\pm${arc['std']:.3f}"

        lines.append(
            f"\\textbf{{\\method{{}}}} & {fid_str} & {lp_str} & {nme_str} & {ss_str} & {arc_str} \\\\"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed evaluation for robust metric reporting"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to ControlNet checkpoint directory"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Directory with test pairs (*_input.png / *_target.png)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/eval_results_multiseed.json",
        help="Output JSON path for aggregated results",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 7],
        help="Random seeds to evaluate (default: 42 123 7)",
    )
    parser.add_argument("--num-steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument(
        "--save-images", action="store_true", help="Save comparison images for each seed"
    )
    args = parser.parse_args()

    print(f"Multi-seed evaluation with seeds: {args.seeds}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test dir: {args.test_dir}")
    print(f"Steps: {args.num_steps}")
    print()

    all_results = []
    for i, seed in enumerate(args.seeds):
        print(f"\n{'=' * 60}")
        print(f"SEED {seed} ({i + 1}/{len(args.seeds)})")
        print(f"{'=' * 60}")
        result = run_single_seed(
            checkpoint=args.checkpoint,
            test_dir=args.test_dir,
            seed=seed,
            num_steps=args.num_steps,
            save_images=args.save_images,
            images_dir=f"paper/predictions_seed{seed}" if args.save_images else None,
        )
        all_results.append(result)

    # Aggregate
    agg = aggregate_seeds(all_results, args.seeds)
    print_summary(agg)

    # LaTeX rows
    print(f"\n{'=' * 60}")
    print("LaTeX Table 1 rows (LandmarkDiff):")
    print(f"{'=' * 60}")
    print(generate_latex_rows(agg))

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregated results saved to {out_path}")


if __name__ == "__main__":
    main()
