"""Extract real surgical displacement vectors from before/after image pairs.

Analyzes paired before/after surgery images to build a data-driven
displacement model that replaces hand-tuned RBF parameters.

Usage:
    python scripts/extract_displacements.py \
        --pairs_dir data/real_surgery_pairs/pairs \
        --output data/displacement_model.npz

The output .npz file contains per-procedure displacement statistics that
can be loaded by DisplacementModel to generate realistic deformations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import PROCEDURE_LANDMARKS


def extract_pair_displacements(
    before_img: np.ndarray,
    after_img: np.ndarray,
    size: int = 512,
) -> dict | None:
    """Extract landmark displacements from a before/after pair.

    Args:
        before_img: BGR before-surgery image.
        after_img: BGR after-surgery image.
        size: Resize both to this size for consistent coordinates.

    Returns:
        Dict with displacement data, or None if face detection fails.
    """
    before = cv2.resize(before_img, (size, size))
    after = cv2.resize(after_img, (size, size))

    face_before = extract_landmarks(before)
    face_after = extract_landmarks(after)

    if face_before is None or face_after is None:
        return None

    coords_before = face_before.landmarks[:, :2].copy()  # (478, 2)
    coords_after = face_after.landmarks[:, :2].copy()

    displacements = coords_after - coords_before  # (478, 2)
    magnitudes = np.linalg.norm(displacements, axis=1)

    # Alignment quality: check that non-surgical landmarks didn't move much
    # Forehead + ears should be stable reference points
    stable_indices = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
        10,
    ]
    stable_mag = magnitudes[stable_indices].mean()

    # Classify procedure by which region has most displacement
    procedure = classify_procedure_from_displacements(displacements, magnitudes)

    quality = max(0.0, 1.0 - stable_mag * 20)  # penalize unstable alignment

    return {
        "landmarks_before": coords_before,
        "landmarks_after": coords_after,
        "displacements": displacements,
        "magnitudes": magnitudes,
        "procedure": procedure,
        "quality_score": float(quality),
        "mean_displacement": float(magnitudes.mean()),
        "max_displacement": float(magnitudes.max()),
    }


def classify_procedure_from_displacements(
    displacements: np.ndarray,
    magnitudes: np.ndarray,
) -> str:
    """Classify which procedure was performed based on displacement patterns.

    Uses PROCEDURE_LANDMARKS to check which anatomical region has most movement.
    """
    scores = {}
    for proc, indices in PROCEDURE_LANDMARKS.items():
        valid = [i for i in indices if i < len(magnitudes)]
        if valid:
            scores[proc] = float(np.mean(magnitudes[valid]))

    if not scores:
        return "unknown"

    best = max(scores, key=scores.get)
    # Only classify if the best region has significantly more displacement
    if scores[best] < 0.005:  # less than 0.5% of image size
        return "unknown"

    return best


def extract_from_directory(
    pairs_dir: Path,
    min_quality: float = 0.3,
) -> list[dict]:
    """Extract displacements from all pairs in a directory.

    Uses _before.png (pre-surgery face) and _target.png (post-surgery face).
    NOTE: _input.png is the MESH rendering, not a face photo -- skip it.
    """
    pairs_dir = Path(pairs_dir)
    results = []
    failed = 0
    seen_prefixes = set()

    # Primary: *_before.png paired with *_target.png or *_after.png
    before_files = sorted(pairs_dir.glob("*_before.png"))
    if before_files:
        print(f"Found {len(before_files)} before-face files")
        for bf in before_files:
            prefix = bf.stem.replace("_before", "")
            if prefix in seen_prefixes:
                continue

            # Try _target.png first (our convention), then _after.png
            target_file = pairs_dir / f"{prefix}_target.png"
            if not target_file.exists():
                target_file = pairs_dir / f"{prefix}_after.png"
            if not target_file.exists():
                continue

            seen_prefixes.add(prefix)
            before_img = cv2.imread(str(bf))
            after_img = cv2.imread(str(target_file))
            if before_img is None or after_img is None:
                failed += 1
                continue

            result = extract_pair_displacements(before_img, after_img)
            if result is not None and result["quality_score"] >= min_quality:
                result["source_file"] = str(bf.name)
                results.append(result)
            else:
                failed += 1

            if (len(results) + failed) % 500 == 0:
                print(
                    f"  Progress: {len(results)} extracted, {failed} failed "
                    f"({len(results) + failed}/{len(before_files)})"
                )

    # Subdirectories
    for subdir in sorted(pairs_dir.iterdir()):
        if not subdir.is_dir():
            continue
        sub_before = sorted(subdir.glob("*_before.png"))
        for bf in sub_before:
            prefix = bf.stem.replace("_before", "")
            for suffix in ["_target.png", "_after.png"]:
                target_file = subdir / f"{prefix}{suffix}"
                if target_file.exists():
                    before_img = cv2.imread(str(bf))
                    after_img = cv2.imread(str(target_file))
                    if before_img is None or after_img is None:
                        failed += 1
                        continue
                    result = extract_pair_displacements(before_img, after_img)
                    if result is not None and result["quality_score"] >= min_quality:
                        result["source_file"] = str(bf.relative_to(pairs_dir))
                        results.append(result)
                    else:
                        failed += 1
                    break

    print(f"\nExtracted {len(results)} valid pairs ({failed} failed/rejected)")
    return results


class DisplacementModel:
    """Data-driven displacement model fitted from real surgical pairs.

    Stores per-procedure, per-landmark displacement statistics and generates
    realistic displacement fields at arbitrary intensity.
    """

    def __init__(self):
        self.procedures: list[str] = []
        self.per_procedure: dict[str, dict] = {}
        self.global_stats: dict = {}

    def fit(self, displacement_list: list[dict]) -> None:
        """Fit model from extracted displacement data.

        Args:
            displacement_list: List of dicts from extract_pair_displacements.
        """
        # Group by procedure
        by_proc: dict[str, list[np.ndarray]] = {}
        for item in displacement_list:
            proc = item["procedure"]
            if proc == "unknown":
                continue
            by_proc.setdefault(proc, []).append(item["displacements"])

        self.procedures = sorted(by_proc.keys())

        for proc, disps in by_proc.items():
            stacked = np.stack(disps)  # (N, 478, 2)
            self.per_procedure[proc] = {
                "mean": stacked.mean(axis=0),  # (478, 2)
                "std": stacked.std(axis=0),  # (478, 2)
                "median": np.median(stacked, axis=0),
                "min": stacked.min(axis=0),
                "max": stacked.max(axis=0),
                "count": len(disps),
                # Per-landmark magnitude stats
                "mag_mean": np.linalg.norm(stacked.mean(axis=0), axis=1),  # (478,)
                "mag_std": np.linalg.norm(stacked.std(axis=0), axis=1),
            }
            print(
                f"  {proc}: {len(disps)} samples, "
                f"mean displacement {self.per_procedure[proc]['mag_mean'].mean():.4f}"
            )

        # Global stats across all procedures
        all_disps = np.concatenate([np.stack(d) for d in by_proc.values()])
        self.global_stats = {
            "mean": all_disps.mean(axis=0),
            "std": all_disps.std(axis=0),
            "total_samples": len(displacement_list),
            "valid_samples": sum(len(d) for d in by_proc.values()),
        }

    def get_displacement_field(
        self,
        procedure: str,
        intensity: float = 50.0,
        noise_std: float = 0.3,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate a displacement field for the given procedure and intensity.

        Args:
            procedure: Procedure name.
            intensity: 0-100 intensity scale.
            noise_std: Per-landmark Gaussian noise (fraction of std).
            rng: Random number generator.

        Returns:
            (478, 2) displacement vectors in normalized coordinates.
        """
        rng = rng or np.random.default_rng()

        if procedure not in self.per_procedure:
            raise ValueError(f"No data for procedure '{procedure}'. Available: {self.procedures}")

        stats = self.per_procedure[procedure]
        scale = intensity / 100.0

        # Base displacement = scaled mean
        base = stats["mean"] * scale

        # Add noise proportional to observed std
        noise = rng.normal(0, noise_std, size=base.shape) * stats["std"] * scale
        field = base + noise

        return field

    def get_rbf_handles(
        self,
        procedure: str,
        intensity: float = 50.0,
        top_k: int = 30,
    ) -> list[tuple[int, np.ndarray, float]]:
        """Convert displacement model to RBF handles for use with existing pipeline.

        Selects the top-K most displaced landmarks as control handles.

        Args:
            procedure: Procedure name.
            intensity: 0-100 intensity scale.
            top_k: Number of control handles to generate.

        Returns:
            List of (landmark_index, displacement_vector, influence_radius) tuples.
        """
        if procedure not in self.per_procedure:
            return []

        stats = self.per_procedure[procedure]
        scale = intensity / 100.0
        mean_disp = stats["mean"] * scale  # (478, 2)
        magnitudes = stats["mag_mean"] * scale  # (478,)

        # Select top-K most displaced landmarks
        top_indices = np.argsort(magnitudes)[-top_k:]

        handles = []
        for idx in top_indices:
            disp = mean_disp[idx]
            mag = magnitudes[idx]
            if mag < 1e-6:
                continue
            # Influence radius proportional to displacement magnitude
            radius = max(15.0, mag * 200)  # in pixels at 512
            handles.append((int(idx), disp, float(radius)))

        return handles

    def save(self, path: str | Path) -> None:
        """Save model to .npz file."""
        path = Path(path)
        data = {"procedures": np.array(self.procedures)}

        for proc, stats in self.per_procedure.items():
            for key, val in stats.items():
                if isinstance(val, np.ndarray):
                    data[f"{proc}_{key}"] = val
                else:
                    data[f"{proc}_{key}"] = np.array(val)

        if self.global_stats:
            for key, val in self.global_stats.items():
                if isinstance(val, np.ndarray):
                    data[f"global_{key}"] = val
                else:
                    data[f"global_{key}"] = np.array(val)

        np.savez_compressed(str(path), **data)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> DisplacementModel:
        """Load model from .npz file."""
        path = Path(path)
        data = np.load(str(path), allow_pickle=True)

        model = cls()
        model.procedures = list(data["procedures"])

        for proc in model.procedures:
            model.per_procedure[proc] = {}
            for key in ["mean", "std", "median", "min", "max", "mag_mean", "mag_std"]:
                k = f"{proc}_{key}"
                if k in data:
                    model.per_procedure[proc][key] = data[k]
            count_key = f"{proc}_count"
            if count_key in data:
                model.per_procedure[proc]["count"] = int(data[count_key])

        for key in ["mean", "std"]:
            gk = f"global_{key}"
            if gk in data:
                model.global_stats[key] = data[gk]
        for key in ["total_samples", "valid_samples"]:
            gk = f"global_{key}"
            if gk in data:
                model.global_stats[key] = int(data[gk])

        return model

    def summary(self) -> str:
        """Print model summary."""
        lines = [f"DisplacementModel: {len(self.procedures)} procedures"]
        for proc in self.procedures:
            stats = self.per_procedure[proc]
            n = stats.get("count", "?")
            mean_mag = stats["mag_mean"].mean() if "mag_mean" in stats else 0
            max_mag = stats["mag_mean"].max() if "mag_mean" in stats else 0
            lines.append(f"  {proc}: {n} samples, mean_mag={mean_mag:.4f}, max_mag={max_mag:.4f}")
        if self.global_stats:
            lines.append(
                f"  Total: {self.global_stats.get('valid_samples', '?')} valid / "
                f"{self.global_stats.get('total_samples', '?')} total"
            )
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Extract surgical displacement vectors")
    parser.add_argument(
        "--pairs_dir", type=str, required=True, help="Directory with before/after pairs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/displacement_model.npz",
        help="Output path for displacement model",
    )
    parser.add_argument(
        "--min_quality", type=float, default=0.3, help="Minimum alignment quality score"
    )
    parser.add_argument("--report", type=str, default=None, help="Path to save JSON report")
    args = parser.parse_args()

    print(f"Extracting displacements from {args.pairs_dir}")

    results = extract_from_directory(Path(args.pairs_dir), min_quality=args.min_quality)

    if not results:
        print("No valid pairs found!")
        sys.exit(1)

    # Fit model
    print(f"\nFitting displacement model from {len(results)} pairs...")
    model = DisplacementModel()
    model.fit(results)

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(f"\n{model.summary()}")

    # Save report
    if args.report:
        report = {
            "total_pairs": len(results),
            "procedures": {},
        }
        for r in results:
            proc = r["procedure"]
            if proc not in report["procedures"]:
                report["procedures"][proc] = {"count": 0, "quality_scores": []}
            report["procedures"][proc]["count"] += 1
            report["procedures"][proc]["quality_scores"].append(r["quality_score"])

        for proc in report["procedures"]:
            qs = report["procedures"][proc]["quality_scores"]
            report["procedures"][proc]["mean_quality"] = float(np.mean(qs))
            report["procedures"][proc]["quality_scores"] = []  # don't bloat JSON

        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
