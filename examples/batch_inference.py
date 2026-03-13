"""Batch inference - process a directory of face images."""

import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Batch LandmarkDiff inference")
    parser.add_argument("input_dir", type=str, help="Directory of input face images")
    parser.add_argument("--procedure", type=str, default="rhinoplasty")
    parser.add_argument("--intensity", type=float, default=0.6)
    parser.add_argument("--output", type=str, default="output/batch/")
    parser.add_argument("--mode", type=str, default="controlnet")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image files
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in extensions)
    print(f"Found {len(images)} images in {input_dir}")

    # Load pipeline once
    from landmarkdiff.inference import LandmarkDiffPipeline
    pipeline = LandmarkDiffPipeline.from_pretrained("checkpoints/latest")

    # Process each image
    results = []
    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] Processing {img_path.name}...")
        try:
            result = pipeline.generate(
                str(img_path),
                procedure=args.procedure,
                intensity=args.intensity,
                mode=args.mode,
            )
            result["prediction"].save(str(output_dir / f"{img_path.stem}_prediction.png"))
            result["comparison"].save(str(output_dir / f"{img_path.stem}_comparison.png"))
            results.append({"file": img_path.name, "status": "success"})
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({"file": img_path.name, "status": "failed", "error": str(e)})

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    print(f"\nDone: {success}/{len(images)} successful")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
