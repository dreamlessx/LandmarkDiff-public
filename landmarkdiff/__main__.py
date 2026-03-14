"""CLI entry point for python -m landmarkdiff."""

import argparse
import sys
from pathlib import Path


def _error(msg: str) -> None:
    """Print error to stderr and exit."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _validate_image_path(path_str: str) -> Path:
    """Validate that the image path exists and looks like an image file."""
    p = Path(path_str)
    if not p.exists():
        _error(f"file not found: {path_str}")
    if not p.is_file():
        _error(f"not a file: {path_str}")
    return p


def main() -> None:
    from landmarkdiff import __version__

    parser = argparse.ArgumentParser(
        prog="landmarkdiff",
        description="Facial surgery outcome prediction from clinical photography",
    )
    parser.add_argument("--version", action="version", version=f"landmarkdiff {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # inference
    infer = subparsers.add_parser("infer", help="Run inference on an image")
    infer.add_argument("image", type=str, help="Path to input face image")
    infer.add_argument(
        "--procedure",
        type=str,
        default="rhinoplasty",
        choices=[
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
        ],
        help="Surgical procedure to simulate (default: rhinoplasty)",
    )
    infer.add_argument(
        "--intensity",
        type=float,
        default=60.0,
        help="Deformation intensity, 0-100 (default: 60)",
    )
    infer.add_argument(
        "--mode",
        type=str,
        default="tps",
        choices=["tps", "controlnet", "img2img", "controlnet_ip"],
        help="Inference mode (default: tps, others require GPU)",
    )
    infer.add_argument(
        "--output",
        type=str,
        default="output/",
        help="Output directory (default: output/)",
    )
    infer.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of diffusion steps (default: 30)",
    )
    infer.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # landmarks
    lm = subparsers.add_parser("landmarks", help="Extract and visualize landmarks")
    lm.add_argument("image", type=str, help="Path to input face image")
    lm.add_argument(
        "--output",
        type=str,
        default="output/landmarks.png",
        help="Output path for landmark visualization (default: output/landmarks.png)",
    )

    # demo
    subparsers.add_parser("demo", help="Launch Gradio web demo")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    try:
        if args.command == "infer":
            _run_inference(args)
        elif args.command == "landmarks":
            _run_landmarks(args)
        elif args.command == "demo":
            _run_demo()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        _error(str(exc))


def _run_inference(args: argparse.Namespace) -> None:
    import numpy as np
    from PIL import Image

    from landmarkdiff.landmarks import extract_landmarks
    from landmarkdiff.manipulation import apply_procedure_preset

    if not (0 <= args.intensity <= 100):
        _error(f"intensity must be between 0 and 100, got {args.intensity}")

    image_path = _validate_image_path(args.image)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGB").resize((512, 512))
    img_array = np.array(img)

    landmarks = extract_landmarks(img_array)
    if landmarks is None:
        _error("no face detected in image")

    deformed = apply_procedure_preset(landmarks, args.procedure, intensity=args.intensity)

    if args.mode == "tps":
        from landmarkdiff.synthetic.tps_warp import warp_image_tps

        src = landmarks.pixel_coords[:, :2].copy()
        dst = deformed.pixel_coords[:, :2].copy()
        src[:, 0] *= 512 / landmarks.image_width
        src[:, 1] *= 512 / landmarks.image_height
        dst[:, 0] *= 512 / deformed.image_width
        dst[:, 1] *= 512 / deformed.image_height
        warped = warp_image_tps(img_array, src, dst)
        Image.fromarray(warped).save(str(output_dir / "prediction.png"))
        print(f"saved tps result to {output_dir / 'prediction.png'}")
    else:
        from landmarkdiff.inference import LandmarkDiffPipeline

        pipeline = LandmarkDiffPipeline(mode=args.mode, device="cuda")
        pipeline.load()
        result = pipeline.generate(
            img_array,
            procedure=args.procedure,
            intensity=args.intensity,
            num_inference_steps=args.steps,
            seed=args.seed,
        )
        result["output"].save(str(output_dir / "prediction.png"))
        print(f"saved result to {output_dir / 'prediction.png'}")


def _run_landmarks(args: argparse.Namespace) -> None:
    import numpy as np
    from PIL import Image

    from landmarkdiff.landmarks import extract_landmarks, render_landmark_image

    image_path = _validate_image_path(args.image)

    img = np.array(Image.open(image_path).convert("RGB").resize((512, 512)))
    landmarks = extract_landmarks(img)
    if landmarks is None:
        _error("no face detected in image")

    mesh = render_landmark_image(landmarks, 512, 512)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(mesh).save(str(output_path))
    print(f"saved landmark mesh to {output_path}")
    print(f"detected {len(landmarks.landmarks)} landmarks, confidence {landmarks.confidence:.2f}")


def _run_demo() -> None:
    try:
        from scripts.app import build_app

        demo = build_app()
        demo.launch()
    except ImportError:
        _error("gradio not installed - run: pip install landmarkdiff[app]")


if __name__ == "__main__":
    main()
