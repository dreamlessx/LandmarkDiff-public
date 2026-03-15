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
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_target_images(data_dir: Path) -> list[Path]:
    """Collect target images (output of pipeline)."""
    # Try *_target.png pattern first (training pairs)
    targets = sorted(data_dir.glob("*_target.png"))
    if targets:
        return targets
    # Try *_output.png pattern (inference output)
    outputs = sorted(data_dir.glob("*_output.png"))
    if outputs:
        return outputs
    # Fall back to all images
    return sorted(f for f in data_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTS and f.is_file())


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


def _try_clean_fid(real_dir: str, gen_dir: str) -> float | None:
    """Attempt FID computation via the clean-fid library (unbiased estimator).

    Returns the FID score on success, or None if clean-fid is not installed.
    """
    try:
        from cleanfid import fid as cfid  # type: ignore[import-untyped]
    except ImportError:
        return None

    score = cfid.compute_fid(real_dir, gen_dir, mode="clean")
    return float(score)


def compute_fid_score(real_dir: str, gen_dir: str) -> dict:
    """Compute FID and optionally IS between two image directories.

    Tries clean-fid first (unbiased estimator), then torch-fidelity,
    then falls back to manual InceptionV3 computation.

    Returns dict with fid, inception_score_mean, inception_score_std,
    real_count, gen_count.
    """
    real_count = len(list(Path(real_dir).glob("*.png")))
    gen_count = len(list(Path(gen_dir).glob("*.png")))

    # Try clean-fid first (unbiased estimator)
    score = _try_clean_fid(real_dir, gen_dir)
    if score is not None:
        logger.info("FID (clean-fid): %.4f  [real=%d, gen=%d]", score, real_count, gen_count)
        return {
            "fid": round(score, 4),
            "inception_score_mean": -1,
            "inception_score_std": -1,
            "real_count": real_count,
            "gen_count": gen_count,
            "method": "clean-fid",
        }

    # Try torch-fidelity
    try:
        from torch_fidelity import calculate_metrics

        metrics = calculate_metrics(
            input1=gen_dir,
            input2=real_dir,
            fid=True,
            isc=True,
            kid=False,
            verbose=False,
        )
        fid_val = round(metrics.get("frechet_inception_distance", -1), 4)
        logger.info("FID (torch-fidelity): %.4f  [real=%d, gen=%d]", fid_val, real_count, gen_count)
        return {
            "fid": fid_val,
            "inception_score_mean": round(metrics.get("inception_score_mean", -1), 4),
            "inception_score_std": round(metrics.get("inception_score_std", -1), 4),
            "real_count": real_count,
            "gen_count": gen_count,
            "method": "torch-fidelity",
        }
    except ImportError:
        pass

    # Fallback: manual FID computation
    logger.info("clean-fid and torch-fidelity not installed; using manual InceptionV3 FID.")
    return _compute_fid_manual(real_dir, gen_dir)


def _compute_fid_manual(real_dir: str, gen_dir: str) -> dict:
    """Manual FID computation using InceptionV3 features.

    Used as fallback when torch-fidelity is not installed.
    """
    import torch
    from torchvision import models, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load InceptionV3 (remove final classification layer)
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False
    )
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

    # Numerical fix: sqrtm can produce complex output due to
    # floating-point rounding on near-singular matrices.
    if np.iscomplexobj(covmean):
        if np.allclose(covmean.imag, 0, atol=1e-3):
            covmean = covmean.real
        else:
            logger.warning(
                "sqrtm produced non-negligible imaginary component "
                "(max imag=%.4e); clipping to real.",
                np.max(np.abs(covmean.imag)),
            )
            covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    fid = max(fid, 0.0)  # FID is non-negative by definition

    return {
        "fid": round(fid, 4),
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
        logger.error("Real image directory not found: %s", real_dir)
        sys.exit(1)

    report = {"real_dir": str(real_dir)}

    if args.compare and len(args.generated) > 1:
        # Compare multiple checkpoints
        logger.info("Comparing %d checkpoints against %s", len(args.generated), real_dir)
        comparisons = {}

        real_imgs = collect_target_images(real_dir)
        logger.info("Real images: %d", len(real_imgs))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            real_tmp = prepare_fid_dir(real_imgs, tmp_path / "real")

            for gen_path_str in args.generated:
                gen_path = Path(gen_path_str)
                if not gen_path.exists():
                    logger.warning("SKIP: %s not found", gen_path)
                    continue

                gen_imgs = collect_target_images(gen_path)
                gen_tmp = prepare_fid_dir(gen_imgs, tmp_path / "gen")
                result = compute_fid_score(str(real_tmp), str(gen_tmp))
                comparisons[str(gen_path)] = result
                logger.info(
                    "  %s: FID=%.2f, IS=%.2f",
                    gen_path.name,
                    result["fid"],
                    result.get("inception_score_mean", -1),
                )

                # Clean gen tmp for next iteration
                shutil.rmtree(gen_tmp)

        report["comparisons"] = comparisons

        # Rank by FID
        ranked = sorted(comparisons.items(), key=lambda x: x[1]["fid"])
        logger.info("\nRanking (by FID, lower is better):")
        for i, (name, metrics) in enumerate(ranked):
            logger.info("  %d. %s: FID=%.2f", i + 1, Path(name).name, metrics["fid"])

    else:
        gen_dir = Path(args.generated[0])
        if not gen_dir.exists():
            logger.error("Generated image directory not found: %s", gen_dir)
            sys.exit(1)

        # Global FID
        logger.info("Computing FID: %s vs %s", gen_dir, real_dir)
        real_imgs = collect_target_images(real_dir)
        gen_imgs = collect_target_images(gen_dir)
        logger.info("Real: %d images, Generated: %d images", len(real_imgs), len(gen_imgs))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            real_tmp = prepare_fid_dir(real_imgs, tmp_path / "real")
            gen_tmp = prepare_fid_dir(gen_imgs, tmp_path / "gen")
            global_result = compute_fid_score(str(real_tmp), str(gen_tmp))

        report["global"] = global_result
        logger.info("\nGlobal FID: %.2f", global_result["fid"])
        if global_result.get("inception_score_mean", -1) > 0:
            logger.info(
                "Inception Score: %.2f +/- %.2f",
                global_result["inception_score_mean"],
                global_result["inception_score_std"],
            )

        # Per-procedure FID
        if args.per_procedure:
            logger.info("\nPer-procedure FID:")
            proc_results = compute_per_procedure_fid(real_dir, gen_dir)
            report["per_procedure"] = proc_results
            for proc, metrics in proc_results.items():
                fid = metrics["fid"]
                n = metrics.get("gen_count", 0)
                logger.info("  %s: FID=%.2f (n=%d)", proc, fid, n)

        # Per-Fitzpatrick FID
        if args.per_fitzpatrick:
            logger.info("\nPer-Fitzpatrick FID:")
            fitz_results = compute_fitzpatrick_fid(real_dir, gen_dir)
            report["per_fitzpatrick"] = fitz_results
            for ftype, metrics in sorted(fitz_results.items()):
                fid = metrics["fid"]
                n_real = metrics.get("real_count", 0)
                n_gen = metrics.get("gen_count", 0)
                logger.info("  Type %s: FID=%.2f (real=%d, gen=%d)", ftype, fid, n_real, n_gen)

    # Save report
    output_path = args.output or str(Path(args.generated[0]) / "fid_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("\nReport saved: %s", output_path)


if __name__ == "__main__":
    main()
