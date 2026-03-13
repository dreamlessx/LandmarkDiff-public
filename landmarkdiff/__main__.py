"""CLI entry point for python -m landmarkdiff."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="landmarkdiff",
        description="Facial surgery outcome prediction from clinical photography",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    subparsers = parser.add_subparsers(dest="command")

    # inference
    infer = subparsers.add_parser("infer", help="Run inference on an image")
    infer.add_argument("image", type=str, help="Path to input face image")
    infer.add_argument("--procedure", type=str, default="rhinoplasty",
                       choices=["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic"])
    infer.add_argument("--intensity", type=float, default=0.6)
    infer.add_argument("--mode", type=str, default="tps",
                       choices=["tps", "controlnet", "img2img", "controlnet_ip"])
    infer.add_argument("--output", type=str, default="output/")
    infer.add_argument("--steps", type=int, default=30)
    infer.add_argument("--seed", type=int, default=None)

    # landmarks
    lm = subparsers.add_parser("landmarks", help="Extract and visualize landmarks")
    lm.add_argument("image", type=str, help="Path to input face image")
    lm.add_argument("--output", type=str, default="output/landmarks.png")

    # demo
    subparsers.add_parser("demo", help="Launch Gradio web demo")

    args = parser.parse_args()

    if args.version:
        from landmarkdiff import __version__
        print(f"landmarkdiff {__version__}")
        return

    if args.command is None:
        parser.print_help()
        return

    if args.command == "infer":
        _run_inference(args)
    elif args.command == "landmarks":
        _run_landmarks(args)
    elif args.command == "demo":
        _run_demo()


def _run_inference(args):
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from landmarkdiff.landmarks import extract_landmarks
    from landmarkdiff.manipulation import apply_procedure_preset

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(args.image).convert("RGB").resize((512, 512))
    img_array = np.array(img)

    landmarks = extract_landmarks(img_array)
    if landmarks is None:
        print("no face detected")
        sys.exit(1)

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


def _run_landmarks(args):
    from pathlib import Path
    import numpy as np
    from landmarkdiff.landmarks import extract_landmarks, render_landmark_image

    landmarks = extract_landmarks(args.image)
    if landmarks is None:
        print("no face detected")
        sys.exit(1)

    mesh = render_landmark_image(landmarks, 512, 512)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    Image.fromarray(mesh).save(str(output_path))
    print(f"saved landmark mesh to {output_path}")
    print(f"detected {len(landmarks.landmarks)} landmarks, confidence {landmarks.confidence:.2f}")


def _run_demo():
    try:
        from scripts.app import create_demo
        demo = create_demo()
        demo.launch()
    except ImportError:
        print("gradio not installed - run: pip install landmarkdiff[app]")
        sys.exit(1)


if __name__ == "__main__":
    main()
