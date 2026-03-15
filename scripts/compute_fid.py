"""Compute FID and Inception Score between image sets.

Standalone script for evaluating generative quality against real surgery
outcome images. Supports per-procedure and per-Fitzpatrick stratification.

Usage:
    # FID between generated outputs and real surgery pairs
    python scripts/compute_fid.py \
        --real data/real_surgery_pairs/pairs \
        --generated results/phaseA_final \
        --output results/fid_report.json

    # Per-procedure FID (expects procedure subdirectories)
    python scripts/compute_fid.py \
        --real data/real_surgery_pairs/pairs \
        --generated results/phaseA_final \
        --per-procedure

    # Compare multiple checkpoints
    python scripts/compute_fid.py \
        --real data/real_surgery_pairs/pairs \
        --generated results/checkpoint-5000 results/checkpoint-10000 results/final \
        --compare
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]


def collect_target_images(data_dir: Path) -> list[Path]:
    """Collect target images (output of pipeline)."""
    extensions = {".jpg", ".jpeg", ".png"}
    # Try *_target.png pattern first (training pairs)
    targets = sorted(data_dir.glob("*_target.png"))
    if targets:
        return targets
    # Try *_output.png pattern (inference output)
    outputs = sorted(data_dir.glob("*_output.png"))
    if outputs:
        return outputs
    # Fall back to all images
    return sorted(f for f in data_dir.rglob("*") if f.suffix.lower() in extensions and f.is_file())


def prepare_fid_dir(images: list[Path], tmp_dir: Path, size: int = 299) -> Path:
    """Copy and resize images to a temp directory for FID computation.

    Inception-v3 expects 299x299 inputs. We resize to avoid resolution bias.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(tmp_dir / f"{i:06d}.png"), img)
    return tmp_dir


def compute_fid_score(real_dir: str, gen_dir: str) -> dict:
    """Compute FID and optionally IS between two image directories.

    Returns dict with fid, inception_score_mean, inception_score_std,
    real_count, gen_count.
    """
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        # Fallback: manual FID computation
        return _compute_fid_manual(real_dir, gen_dir)

    metrics = calculate_metrics(
        input1=gen_dir,
        input2=real_dir,
        fid=True,
        isc=True,
        kid=False,
        verbose=False,
    )

    real_count = len(list(Path(real_dir).glob("*.png")))
    gen_count = len(list(Path(gen_dir).glob("*.png")))

    return {
        "fid": round(metrics.get("frechet_inception_distance", -1), 4),
        "inception_score_mean": round(metrics.get("inception_score_mean", -1), 4),
        "inception_score_std": round(metrics.get("inception_score_std", -1), 4),
        "real_count": real_count,
        "gen_count": gen_count,
    }


def _compute_fid_manual(real_dir: str, gen_dir: str) -> dict:
    """Manual FID computation using InceptionV3 features.

    Used as fallback when torch-fidelity is not installed.
    """
    import torch
    from torchvision import models, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load InceptionV3 (remove final classification layer)
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()
    inception.eval().to(device)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def extract_features(img_dir: str) -> np.ndarray:
        features = []
        for img_path in sorted(Path(img_dir).glob("*.png")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(img_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = inception(tensor).cpu().numpy().flatten()
            features.append(feat)
        return np.array(features)

    real_feats = extract_features(real_dir)
    gen_feats = extract_features(gen_dir)

    if len(real_feats) < 2 or len(gen_feats) < 2:
        return {
            "fid": -1,
            "error": "Not enough images",
            "real_count": len(real_feats),
            "gen_count": len(gen_feats),
        }

    # Compute FID
    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_g, sigma_g = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)

    from scipy.linalg import sqrtm

    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)

    return {
        "fid": round(float(fid), 4),
        "inception_score_mean": -1,
        "inception_score_std": -1,
        "real_count": len(real_feats),
        "gen_count": len(gen_feats),
        "method": "manual",
    }


def compute_per_procedure_fid(
    real_dir: Path,
    gen_dir: Path,
) -> dict[str, dict]:
    """Compute FID per procedure.

    Expects procedure-named subdirectories or file prefixes.
    """
    results = {}
    for proc in PROCEDURES:
        # Try subdirectory first
        real_proc = real_dir / proc
        gen_proc = gen_dir / proc

        if not real_proc.exists():
            # Try prefix-based filtering
            real_imgs = sorted(real_dir.glob(f"{proc}_*_target.png"))
            gen_imgs = sorted(gen_dir.glob(f"{proc}_*_output.png"))
            if not gen_imgs:
                gen_imgs = sorted(gen_dir.glob(f"{proc}_*_target.png"))
        else:
            real_imgs = collect_target_images(real_proc)
            gen_imgs = collect_target_images(gen_proc)

        if len(real_imgs) < 2 or len(gen_imgs) < 2:
            results[proc] = {
                "fid": -1,
                "error": "insufficient images",
                "real_count": len(real_imgs) if not real_proc.exists() else 0,
                "gen_count": len(gen_imgs) if not gen_proc.exists() else 0,
            }
            continue

        # Prepare temp dirs
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            real_tmp = prepare_fid_dir(real_imgs, tmp_path / "real")
            gen_tmp = prepare_fid_dir(gen_imgs, tmp_path / "gen")
            results[proc] = compute_fid_score(str(real_tmp), str(gen_tmp))

    return results


def compute_fitzpatrick_fid(
    real_dir: Path,
    gen_dir: Path,
) -> dict[str, dict]:
    """Compute FID stratified by Fitzpatrick skin type."""
    from landmarkdiff.evaluation import classify_fitzpatrick_ita

    def classify_images(img_dir: Path) -> dict[str, list[Path]]:
        groups: dict[str, list[Path]] = {}
        for img_path in sorted(img_dir.rglob("*.png")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            fitz = classify_fitzpatrick_ita(img)
            groups.setdefault(fitz, []).append(img_path)
        return groups

    real_groups = classify_images(real_dir)
    gen_groups = classify_images(gen_dir)

    results = {}
    for ftype in sorted(set(list(real_groups.keys()) + list(gen_groups.keys()))):
        real_imgs = real_groups.get(ftype, [])
        gen_imgs = gen_groups.get(ftype, [])
        if len(real_imgs) < 2 or len(gen_imgs) < 2:
            results[ftype] = {"fid": -1, "real_count": len(real_imgs), "gen_count": len(gen_imgs)}
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            real_tmp = prepare_fid_dir(real_imgs, tmp_path / "real")
            gen_tmp = prepare_fid_dir(gen_imgs, tmp_path / "gen")
            results[ftype] = compute_fid_score(str(real_tmp), str(gen_tmp))

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute FID and IS metrics")
    parser.add_argument("--real", required=True, help="Directory of real images")
    parser.add_argument(
        "--generated", nargs="+", required=True, help="Directory/directories of generated images"
    )
    parser.add_argument("--output", default=None, help="Output JSON report path")
    parser.add_argument("--per-procedure", action="store_true", help="Compute FID per procedure")
    parser.add_argument(
        "--per-fitzpatrick", action="store_true", help="Compute FID per Fitzpatrick skin type"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare multiple generated directories"
    )
    args = parser.parse_args()

    real_dir = Path(args.real)
    if not real_dir.exists():
        print(f"ERROR: Real image directory not found: {real_dir}")
        sys.exit(1)

    report = {"real_dir": str(real_dir)}

    if args.compare and len(args.generated) > 1:
        # Compare multiple checkpoints
        print(f"Comparing {len(args.generated)} checkpoints against {real_dir}")
        comparisons = {}

        real_imgs = collect_target_images(real_dir)
        print(f"Real images: {len(real_imgs)}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            real_tmp = prepare_fid_dir(real_imgs, tmp_path / "real")

            for gen_path_str in args.generated:
                gen_path = Path(gen_path_str)
                if not gen_path.exists():
                    print(f"  SKIP: {gen_path} not found")
                    continue

                gen_imgs = collect_target_images(gen_path)
                gen_tmp = prepare_fid_dir(gen_imgs, tmp_path / "gen")
                result = compute_fid_score(str(real_tmp), str(gen_tmp))
                comparisons[str(gen_path)] = result
                print(
                    f"  {gen_path.name}: FID={result['fid']:.2f}, "
                    f"IS={result.get('inception_score_mean', -1):.2f}"
                )

                # Clean gen tmp for next iteration
                shutil.rmtree(gen_tmp)

        report["comparisons"] = comparisons

        # Rank by FID
        ranked = sorted(comparisons.items(), key=lambda x: x[1]["fid"])
        print("\nRanking (by FID, lower is better):")
        for i, (name, metrics) in enumerate(ranked):
            print(f"  {i + 1}. {Path(name).name}: FID={metrics['fid']:.2f}")

    else:
        gen_dir = Path(args.generated[0])
        if not gen_dir.exists():
            print(f"ERROR: Generated image directory not found: {gen_dir}")
            sys.exit(1)

        # Global FID
        print(f"Computing FID: {gen_dir} vs {real_dir}")
        real_imgs = collect_target_images(real_dir)
        gen_imgs = collect_target_images(gen_dir)
        print(f"Real: {len(real_imgs)} images, Generated: {len(gen_imgs)} images")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            real_tmp = prepare_fid_dir(real_imgs, tmp_path / "real")
            gen_tmp = prepare_fid_dir(gen_imgs, tmp_path / "gen")
            global_result = compute_fid_score(str(real_tmp), str(gen_tmp))

        report["global"] = global_result
        print(f"\nGlobal FID: {global_result['fid']:.2f}")
        if global_result.get("inception_score_mean", -1) > 0:
            print(
                f"Inception Score: {global_result['inception_score_mean']:.2f} "
                f"+/- {global_result['inception_score_std']:.2f}"
            )

        # Per-procedure FID
        if args.per_procedure:
            print("\nPer-procedure FID:")
            proc_results = compute_per_procedure_fid(real_dir, gen_dir)
            report["per_procedure"] = proc_results
            for proc, metrics in proc_results.items():
                fid = metrics["fid"]
                n = metrics.get("gen_count", 0)
                print(f"  {proc}: FID={fid:.2f} (n={n})")

        # Per-Fitzpatrick FID
        if args.per_fitzpatrick:
            print("\nPer-Fitzpatrick FID:")
            fitz_results = compute_fitzpatrick_fid(real_dir, gen_dir)
            report["per_fitzpatrick"] = fitz_results
            for ftype, metrics in sorted(fitz_results.items()):
                fid = metrics["fid"]
                n_real = metrics.get("real_count", 0)
                n_gen = metrics.get("gen_count", 0)
                print(f"  Type {ftype}: FID={fid:.2f} (real={n_real}, gen={n_gen})")

    # Save report
    output_path = args.output or str(Path(args.generated[0]) / "fid_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()
