"""CLI for face distortion detection and neural restoration (single or batch)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.face_verifier import (
    analyze_distortions,
    neural_quality_score,
    verify_and_restore,
    verify_batch,
)


def process_single(args):
    """Process a single face image."""
    image = cv2.imread(args.input)
    if image is None:
        print(f"ERROR: Could not load {args.input}")
        sys.exit(1)

    image = cv2.resize(image, (512, 512))
    print(f"Loaded: {args.input} ({image.shape})")

    if args.analyze_only:
        report = analyze_distortions(image)
        print(f"\n{report.summary()}")

        quality = neural_quality_score(image)
        print(f"\nNeural Quality Score: {quality:.1f}/100")
        return

    print("\nRunning verification + restoration pipeline...")
    result = verify_and_restore(
        image,
        quality_threshold=args.quality_threshold,
        identity_threshold=args.identity_threshold,
        restore_mode=args.restore_mode,
        codeformer_fidelity=args.codeformer_fidelity,
    )

    print(f"\n{result.summary()}")
    print(f"\nDistortion Analysis:\n{result.distortion_report.summary()}")

    # Save outputs
    out = args.output or "fixed_" + Path(args.input).name
    cv2.imwrite(out, result.restored)
    print(f"\nSaved restored image: {out}")

    # Save side-by-side comparison
    if args.comparison:
        h, w = image.shape[:2]
        restored_resized = cv2.resize(result.restored, (w, h))
        comparison = np.hstack([image, restored_resized])
        comp_path = str(Path(out).with_suffix("")) + "_comparison.png"
        cv2.imwrite(comp_path, comparison)
        print(f"Saved comparison:     {comp_path}")

    # Save JSON report
    if args.json_report:
        report_data = {
            "input": args.input,
            "output": out,
            "quality_before": result.distortion_report.quality_score,
            "quality_after": result.post_quality_score,
            "improvement": result.improvement,
            "identity_similarity": result.identity_similarity,
            "identity_preserved": result.identity_preserved,
            "primary_distortion": result.distortion_report.primary_distortion,
            "severity": result.distortion_report.severity,
            "stages": result.restoration_stages,
            "distortions": {
                "blur": result.distortion_report.blur_score,
                "noise": result.distortion_report.noise_score,
                "compression": result.distortion_report.compression_score,
                "oversmooth": result.distortion_report.oversmooth_score,
                "color_cast": result.distortion_report.color_cast_score,
                "geometric": result.distortion_report.geometric_distort,
                "lighting": result.distortion_report.lighting_score,
            },
        }
        json_path = str(Path(out).with_suffix("")) + "_report.json"
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"Saved JSON report:    {json_path}")


def process_batch(args):
    """Process a directory of face images."""
    print(f"Batch processing: {args.input}")
    print(f"Quality threshold: {args.quality_threshold}")
    print(f"Identity threshold: {args.identity_threshold}")
    print(f"Restore mode: {args.restore_mode}")
    print()

    report = verify_batch(
        image_dir=args.input,
        output_dir=args.output,
        quality_threshold=args.quality_threshold,
        identity_threshold=args.identity_threshold,
        restore_mode=args.restore_mode,
        save_rejected=args.save_rejected,
    )

    # Save JSON report
    if args.json_report:
        out_dir = args.output or str(Path(args.input).parent / f"{Path(args.input).name}_verified")
        json_path = str(Path(out_dir) / "report.json")
        report_data = {
            "total": report.total,
            "passed": report.passed,
            "restored": report.restored,
            "rejected": report.rejected,
            "identity_failures": report.identity_failures,
            "avg_quality_before": report.avg_quality_before,
            "avg_quality_after": report.avg_quality_after,
            "avg_identity_sim": report.avg_identity_sim,
            "distortion_counts": report.distortion_counts,
        }
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\nJSON report: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face verification, distortion detection, and neural restoration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Image path or directory (with --batch)")
    parser.add_argument("--output", "-o", help="Output path (file or directory)")
    parser.add_argument("--batch", action="store_true", help="Process entire directory")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't restore")
    parser.add_argument(
        "--comparison", action="store_true", default=True, help="Save side-by-side comparison"
    )
    parser.add_argument("--json-report", action="store_true", default=True, help="Save JSON report")
    parser.add_argument(
        "--save-rejected", action="store_true", help="Save rejected images in batch mode"
    )

    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=60.0,
        help="Min quality to skip restoration (0-100, default: 60)",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=0.6,
        help="Min ArcFace similarity to pass (0-1, default: 0.6)",
    )
    parser.add_argument(
        "--restore-mode",
        default="auto",
        choices=["auto", "codeformer", "gfpgan", "all"],
        help="Restoration mode (default: auto)",
    )
    parser.add_argument(
        "--codeformer-fidelity",
        type=float,
        default=0.7,
        help="CodeFormer quality-fidelity balance (0=quality, 1=fidelity)",
    )

    args = parser.parse_args()

    if args.batch:
        process_batch(args)
    else:
        process_single(args)
