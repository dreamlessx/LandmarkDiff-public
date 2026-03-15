"""Unified CLI for LandmarkDiff.

Usage:
    landmarkdiff infer IMAGE --procedure rhinoplasty --intensity 65
    landmarkdiff evaluate --test-dir data/test --checkpoint checkpoints/latest
    landmarkdiff train --config configs/phaseA.yaml
    landmarkdiff demo IMAGE --output demo_report.png
    landmarkdiff config --show
    landmarkdiff validate IMAGE --output validated.png
"""

from __future__ import annotations

import argparse
import sys


def cmd_infer(args: argparse.Namespace) -> None:
    """Run single-image inference."""
    from pathlib import Path

    import cv2

    from landmarkdiff.inference import LandmarkDiffPipeline

    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Cannot read image: {args.image}")
        sys.exit(1)

    image = cv2.resize(image, (512, 512))

    pipeline = LandmarkDiffPipeline(
        mode=args.mode,
        controlnet_checkpoint=args.checkpoint,
        displacement_model_path=args.displacement_model,
    )
    pipeline.load()

    result = pipeline.generate(
        image,
        procedure=args.procedure,
        intensity=args.intensity,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result["output"])
    print(f"Output saved: {out_path}")

    if args.watermark:
        from landmarkdiff.safety import SafetyValidator

        validator = SafetyValidator()
        watermarked = validator.apply_watermark(result["output"])
        wm_path = out_path.with_stem(out_path.stem + "_watermarked")
        cv2.imwrite(str(wm_path), watermarked)
        print(f"Watermarked: {wm_path}")


def cmd_ensemble(args: argparse.Namespace) -> None:
    """Run ensemble inference."""
    from landmarkdiff.ensemble import ensemble_inference

    ensemble_inference(
        image_path=args.image,
        procedure=args.procedure,
        intensity=args.intensity,
        output_dir=args.output,
        n_samples=args.n_samples,
        strategy=args.strategy,
        mode=args.mode,
        controlnet_checkpoint=args.checkpoint,
        displacement_model_path=args.displacement_model,
        seed=args.seed,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation on test set."""
    from pathlib import Path

    # Import evaluation functions
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.run_evaluation import run_evaluation

    run_evaluation(
        test_dir=args.test_dir,
        output_dir=args.output,
        checkpoint=args.checkpoint,
        max_samples=args.max_samples,
    )


def cmd_config(args: argparse.Namespace) -> None:
    """Show or validate configuration."""
    from landmarkdiff.config import ExperimentConfig, load_config, validate_config

    config = load_config(args.file) if args.file else ExperimentConfig()

    if args.validate:
        warnings = validate_config(config)
        if warnings:
            print("Validation warnings:")
            for w in warnings:
                print(f"  - {w}")
        else:
            print("Configuration valid (no warnings).")
    else:
        from dataclasses import asdict

        import yaml

        print(yaml.dump(asdict(config), default_flow_style=False, sort_keys=False))


def cmd_validate(args: argparse.Namespace) -> None:
    """Run safety validation on an output image."""
    import cv2

    from landmarkdiff.safety import SafetyValidator

    input_img = cv2.imread(args.input)
    output_img = cv2.imread(args.output_image)

    if input_img is None or output_img is None:
        print("ERROR: Cannot read input or output image.")
        sys.exit(1)

    validator = SafetyValidator(
        watermark_enabled=args.watermark,
    )

    result = validator.validate(
        input_image=input_img,
        output_image=output_img,
        face_confidence=args.face_confidence,
    )

    print(result.summary())

    if not result.passed:
        sys.exit(1)


def cmd_version(args: argparse.Namespace) -> None:
    """Print version info."""
    from landmarkdiff import __version__

    print(f"LandmarkDiff v{__version__}")


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="landmarkdiff",
        description="LandmarkDiff: Facial surgery outcome prediction via latent diffusion",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- infer ---
    p_infer = subparsers.add_parser("infer", help="Run single-image inference")
    p_infer.add_argument("image", help="Input face image path")
    p_infer.add_argument(
        "--procedure",
        default="rhinoplasty",
        choices=[
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
        ],
    )
    p_infer.add_argument("--intensity", type=float, default=65.0)
    p_infer.add_argument("--output", default="output.png")
    p_infer.add_argument("--mode", default="tps", choices=["controlnet", "img2img", "tps"])
    p_infer.add_argument("--checkpoint", default=None)
    p_infer.add_argument("--displacement-model", default=None)
    p_infer.add_argument("--seed", type=int, default=42)
    p_infer.add_argument("--watermark", action="store_true")
    p_infer.set_defaults(func=cmd_infer)

    # --- ensemble ---
    p_ensemble = subparsers.add_parser("ensemble", help="Run ensemble inference")
    p_ensemble.add_argument("image", help="Input face image path")
    p_ensemble.add_argument("--procedure", default="rhinoplasty")
    p_ensemble.add_argument("--intensity", type=float, default=65.0)
    p_ensemble.add_argument("--output", default="ensemble_output")
    p_ensemble.add_argument("--n-samples", type=int, default=5)
    p_ensemble.add_argument(
        "--strategy",
        default="best_of_n",
        choices=["pixel_average", "weighted_average", "best_of_n", "median"],
    )
    p_ensemble.add_argument("--mode", default="tps", choices=["controlnet", "img2img", "tps"])
    p_ensemble.add_argument("--checkpoint", default=None)
    p_ensemble.add_argument("--displacement-model", default=None)
    p_ensemble.add_argument("--seed", type=int, default=42)
    p_ensemble.set_defaults(func=cmd_ensemble)

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Evaluate on test set")
    p_eval.add_argument("--test-dir", required=True)
    p_eval.add_argument("--output", default="eval_results")
    p_eval.add_argument("--mode", default="tps")
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--displacement-model", default=None)
    p_eval.add_argument("--max-samples", type=int, default=0)
    p_eval.set_defaults(func=cmd_evaluate)

    # --- config ---
    p_config = subparsers.add_parser("config", help="Show or validate configuration")
    p_config.add_argument("--file", default=None, help="YAML config file")
    p_config.add_argument("--validate", action="store_true")
    p_config.set_defaults(func=cmd_config)

    # --- validate ---
    p_validate = subparsers.add_parser("validate", help="Run safety validation")
    p_validate.add_argument("input", help="Original input image")
    p_validate.add_argument("output_image", help="Generated output image")
    p_validate.add_argument("--watermark", action="store_true")
    p_validate.add_argument("--face-confidence", type=float, default=1.0)
    p_validate.set_defaults(func=cmd_validate)

    # --- version ---
    p_version = subparsers.add_parser("version", help="Print version")
    p_version.set_defaults(func=cmd_version)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
