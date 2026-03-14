#!/usr/bin/env python3
"""Evaluate LandmarkDiff output quality against ground-truth targets.

Computes:
- SSIM (structural similarity) between original and prediction
- ArcFace identity preservation score (cosine similarity)
- Landmark accuracy (NME: normalized mean error, deformed vs target)
- LPIPS perceptual distance
- FID (if a dataset directory is provided)

Supports per-procedure and per-Fitzpatrick-type stratification.

Usage:
    # Evaluate a single prediction against its target
    python scripts/evaluate_quality.py \
        --pred output.png --target target.png --original input.png

    # Evaluate a directory of prediction/target pairs
    python scripts/evaluate_quality.py \
        --pred_dir results/predictions --target_dir results/targets \
        --output results/quality_report

    # Compute FID between two image directories
    python scripts/evaluate_quality.py \
        --pred_dir results/generated --target_dir data/real \
        --compute-fid --output results/quality_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Metric wrappers
# ---------------------------------------------------------------------------


def compute_ssim_score(pred: np.ndarray, target: np.ndarray) -> float:
    """SSIM between two images. Returns value in [0, 1]."""
    from landmarkdiff.evaluation import compute_ssim

    return compute_ssim(pred, target)


def compute_identity_score(pred: np.ndarray, target: np.ndarray) -> float:
    """ArcFace cosine similarity between two face images.

    Returns similarity in [0, 1]. Falls back to SSIM if InsightFace
    is not installed.
    """
    from landmarkdiff.evaluation import compute_identity_similarity

    return compute_identity_similarity(pred, target)


def compute_landmark_nme(pred: np.ndarray, target: np.ndarray) -> float:
    """Normalized Mean Error between detected landmarks.

    Extracts landmarks from both images and computes NME normalized
    by inter-ocular distance.
    """
    from landmarkdiff.evaluation import compute_nme
    from landmarkdiff.landmarks import extract_landmarks

    pred_face = extract_landmarks(pred)
    target_face = extract_landmarks(target)
    if pred_face is None or target_face is None:
        return float("nan")
    return compute_nme(pred_face.pixel_coords, target_face.pixel_coords)


def compute_lpips_score(pred: np.ndarray, target: np.ndarray) -> float:
    """LPIPS perceptual distance. Lower is better."""
    from landmarkdiff.evaluation import compute_lpips

    val = compute_lpips(pred, target)
    return val if val is not None else float("nan")


def compute_fid_score(pred_dir: str, target_dir: str) -> float:
    """FID between two directories of images."""
    try:
        from landmarkdiff.fid import compute_fid_from_dirs

        return compute_fid_from_dirs(pred_dir, target_dir)
    except ImportError:
        from landmarkdiff.evaluation import compute_fid

        return compute_fid(target_dir, pred_dir)


def classify_skin_type(image: np.ndarray) -> str:
    """Classify Fitzpatrick skin type from an image via ITA."""
    try:
        from landmarkdiff.evaluation import classify_fitzpatrick_ita

        return classify_fitzpatrick_ita(image)
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# Single-pair evaluation
# ---------------------------------------------------------------------------


def evaluate_single(
    pred: np.ndarray,
    target: np.ndarray,
    original: np.ndarray | None = None,
    compute_identity: bool = True,
) -> dict:
    """Evaluate a single prediction/target pair."""
    result: dict = {}

    result["ssim"] = round(compute_ssim_score(pred, target), 4)
    result["lpips"] = round(compute_lpips_score(pred, target), 4)
    result["nme"] = round(compute_landmark_nme(pred, target), 4)

    if compute_identity and original is not None:
        result["identity_sim"] = round(compute_identity_score(pred, original), 4)
    elif compute_identity:
        result["identity_sim"] = round(compute_identity_score(pred, target), 4)

    result["fitzpatrick"] = classify_skin_type(target)

    return result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def load_paired_images(
    pred_dir: str,
    target_dir: str,
    original_dir: str | None = None,
    max_samples: int = 0,
) -> list[dict]:
    """Load paired images from directories.

    Matches files by name stem. Supports common extensions.
    """
    pred_path = Path(pred_dir)
    target_path = Path(target_dir)

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    pred_files = {f.stem: f for f in sorted(pred_path.iterdir()) if f.suffix.lower() in exts}
    target_files = {f.stem: f for f in sorted(target_path.iterdir()) if f.suffix.lower() in exts}

    orig_files = {}
    if original_dir:
        orig_path = Path(original_dir)
        orig_files = {f.stem: f for f in sorted(orig_path.iterdir()) if f.suffix.lower() in exts}

    common_stems = sorted(set(pred_files) & set(target_files))
    if max_samples > 0:
        common_stems = common_stems[:max_samples]

    pairs = []
    for stem in common_stems:
        entry = {
            "stem": stem,
            "pred_path": str(pred_files[stem]),
            "target_path": str(target_files[stem]),
        }
        if stem in orig_files:
            entry["original_path"] = str(orig_files[stem])

        # Try to infer procedure from filename
        proc = "unknown"
        for p in [
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
        ]:
            if p in stem:
                proc = p
                break
        entry["procedure"] = proc
        pairs.append(entry)

    return pairs


def evaluate_batch(
    pairs: list[dict],
    compute_identity: bool = True,
) -> dict:
    """Evaluate all pairs and compute aggregate + stratified metrics."""
    per_sample = []
    proc_groups: dict[str, list[dict]] = {}
    fitz_groups: dict[str, list[dict]] = {}

    for i, pair in enumerate(pairs):
        pred = cv2.imread(pair["pred_path"])
        target = cv2.imread(pair["target_path"])
        if pred is None or target is None:
            print(f"  Skip {pair['stem']}: cannot read image(s)")
            continue

        # Resize to match
        if pred.shape[:2] != target.shape[:2]:
            pred = cv2.resize(pred, (target.shape[1], target.shape[0]))

        original = None
        if "original_path" in pair:
            original = cv2.imread(pair["original_path"])
            if original is not None and original.shape[:2] != target.shape[:2]:
                original = cv2.resize(original, (target.shape[1], target.shape[0]))

        metrics = evaluate_single(pred, target, original, compute_identity)
        metrics["stem"] = pair["stem"]
        metrics["procedure"] = pair["procedure"]
        per_sample.append(metrics)

        proc_groups.setdefault(pair["procedure"], []).append(metrics)
        fitz = metrics.get("fitzpatrick", "?")
        if fitz != "?":
            fitz_groups.setdefault(fitz, []).append(metrics)

        if (i + 1) % 25 == 0 or (i + 1) == len(pairs):
            print(f"  [{i + 1}/{len(pairs)}] evaluated")

    if not per_sample:
        return {"error": "No valid pairs evaluated"}

    # Aggregate
    def _agg(samples: list[dict], key: str) -> dict:
        vals = [s[key] for s in samples if not np.isnan(s.get(key, float("nan")))]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "n": len(vals),
        }

    metric_keys = ["ssim", "lpips", "nme"]
    if compute_identity:
        metric_keys.append("identity_sim")

    aggregate = {k: _agg(per_sample, k) for k in metric_keys}

    # Per-procedure
    per_procedure = {}
    for proc, samples in proc_groups.items():
        per_procedure[proc] = {k: _agg(samples, k) for k in metric_keys}

    # Per-Fitzpatrick
    per_fitzpatrick: dict[str, dict[str, Any]] = {}
    for ftype, samples in fitz_groups.items():
        per_fitzpatrick[ftype] = {k: _agg(samples, k) for k in metric_keys}
        per_fitzpatrick[ftype]["count"] = len(samples)

    return {
        "aggregate": aggregate,
        "per_procedure": per_procedure,
        "per_fitzpatrick": per_fitzpatrick,
        "per_sample": per_sample,
        "total_evaluated": len(per_sample),
    }


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def format_summary(results: dict) -> str:
    """Format results as a readable summary."""
    lines = ["=" * 60, "Quality Evaluation Results", "=" * 60, ""]

    if "error" in results:
        lines.append(f"ERROR: {results['error']}")
        return "\n".join(lines)

    agg = results["aggregate"]
    lines.append("Aggregate Metrics:")
    for k, v in agg.items():
        arrow = "(higher is better)" if k in ("ssim", "identity_sim") else "(lower is better)"
        lines.append(f"  {k:>15s}: {v['mean']:.4f} +/- {v['std']:.4f}  (n={v['n']})  {arrow}")

    if results.get("per_procedure"):
        lines.append("")
        lines.append("Per-Procedure Breakdown:")
        for proc, metrics in sorted(results["per_procedure"].items()):
            parts = [f"{proc}:"]
            for k, v in metrics.items():
                parts.append(f"{k}={v['mean']:.4f}")
            lines.append("  " + " ".join(parts))

    if results.get("per_fitzpatrick"):
        lines.append("")
        lines.append("Per-Fitzpatrick Breakdown:")
        for ftype in sorted(results["per_fitzpatrick"]):
            entry = results["per_fitzpatrick"][ftype]
            n = entry.get("count", 0)
            parts = [f"Type {ftype} (n={n}):"]
            for k, v in entry.items():
                if k == "count":
                    continue
                parts.append(f"{k}={v['mean']:.4f}")
            lines.append("  " + " ".join(parts))

    if "fid" in results:
        lines.append(f"\nFID: {results['fid']:.2f}")

    lines.append("")
    lines.append(f"Total pairs evaluated: {results.get('total_evaluated', 0)}")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_markdown_table(results: dict) -> str:
    """Format aggregate results as markdown."""
    lines = [
        "## Quality Metrics",
        "",
        "| Metric | Mean | Std | N |",
        "|--------|------|-----|---|",
    ]
    for k, v in results.get("aggregate", {}).items():
        lines.append(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} | {v['n']} |")

    if results.get("per_procedure"):
        lines.extend(["", "## Per-Procedure", ""])
        procs = sorted(results["per_procedure"])
        header = "| Metric | " + " | ".join(procs) + " |"
        sep = "|--------|" + "|".join(["------"] * len(procs)) + "|"
        lines.extend([header, sep])
        metric_keys = list(next(iter(results["per_procedure"].values())).keys())
        for mk in metric_keys:
            row = f"| {mk} |"
            for p in procs:
                v = results["per_procedure"][p].get(mk, {})
                row += f" {v.get('mean', 0):.4f} |"
            lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate LandmarkDiff output quality")

    # Single-pair mode
    parser.add_argument("--pred", default=None, help="Single prediction image")
    parser.add_argument("--target", default=None, help="Single target image")
    parser.add_argument("--original", default=None, help="Original input image")

    # Batch mode
    parser.add_argument("--pred_dir", default=None, help="Directory of predictions")
    parser.add_argument("--target_dir", default=None, help="Directory of targets")
    parser.add_argument("--original_dir", default=None, help="Directory of originals")

    # Options
    parser.add_argument("--max_samples", type=int, default=0, help="Limit samples")
    parser.add_argument(
        "--compute-fid", action="store_true", help="Compute FID (needs directories)"
    )
    parser.add_argument("--no-identity", action="store_true", help="Skip ArcFace identity check")
    parser.add_argument("--output", default="results/quality", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    compute_identity = not args.no_identity

    # Single-pair mode
    if args.pred and args.target:
        pred = cv2.imread(args.pred)
        target = cv2.imread(args.target)
        if pred is None:
            print(f"ERROR: Cannot read {args.pred}")
            sys.exit(1)
        if target is None:
            print(f"ERROR: Cannot read {args.target}")
            sys.exit(1)

        original = None
        if args.original:
            original = cv2.imread(args.original)

        # Match sizes
        if pred.shape[:2] != target.shape[:2]:
            pred = cv2.resize(pred, (target.shape[1], target.shape[0]))
        if original is not None and original.shape[:2] != target.shape[:2]:
            original = cv2.resize(original, (target.shape[1], target.shape[0]))

        metrics = evaluate_single(pred, target, original, compute_identity)
        print("Single-Pair Evaluation:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        with open(output_dir / "quality_single.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to {output_dir / 'quality_single.json'}")
        return

    # Batch mode
    if args.pred_dir and args.target_dir:
        print(f"Loading pairs from {args.pred_dir} and {args.target_dir}...")
        pairs = load_paired_images(
            args.pred_dir,
            args.target_dir,
            args.original_dir,
            args.max_samples,
        )
        print(f"Found {len(pairs)} matched pairs")

        if not pairs:
            print("No matched pairs found. Check that filenames match between dirs.")
            sys.exit(1)

        results = evaluate_batch(pairs, compute_identity)

        # FID
        if args.compute_fid:
            print("\nComputing FID...")
            try:
                fid = compute_fid_score(args.pred_dir, args.target_dir)
                results["fid"] = round(fid, 2)
                print(f"  FID: {fid:.2f}")
            except Exception as e:
                print(f"  FID computation failed: {e}")
                results["fid_error"] = str(e)

        # Print summary
        summary = format_summary(results)
        print()
        print(summary)

        # Save outputs
        json_path = output_dir / "quality_results.json"
        # Remove per_sample from JSON if too large (keep aggregate)
        save_results = {k: v for k, v in results.items() if k != "per_sample"}
        save_results["sample_count"] = len(results.get("per_sample", []))
        with open(json_path, "w") as f:
            json.dump(save_results, f, indent=2)

        # Full per-sample results
        per_sample_path = output_dir / "quality_per_sample.json"
        with open(per_sample_path, "w") as f:
            json.dump(results.get("per_sample", []), f, indent=2)

        md_path = output_dir / "quality_results.md"
        md_path.write_text(format_markdown_table(results))

        print(f"\nResults saved to {output_dir}/")
        print(f"  {json_path.name} -- aggregate + stratified metrics")
        print(f"  {per_sample_path.name} -- per-sample breakdown")
        print(f"  {md_path.name} -- markdown table")
        return

    parser.print_help()
    print("\nProvide either --pred/--target for single evaluation or")
    print("--pred_dir/--target_dir for batch evaluation.")
    sys.exit(1)


if __name__ == "__main__":
    main()
