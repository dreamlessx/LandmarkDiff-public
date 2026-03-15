"""Run ControlNet inference and produce before/after results."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.inference import LandmarkDiffPipeline
from landmarkdiff.masking import mask_to_3channel


def run_single(
    pipe: LandmarkDiffPipeline,
    image_path: str,
    procedure: str,
    intensity: float,
    output_dir: Path,
    seed: int = 42,
) -> bool:
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: Could not load {image_path}")
        return False

    image_512 = cv2.resize(image, (512, 512))

    t0 = time.time()
    try:
        result = pipe.generate(
            image_512,
            procedure=procedure,
            intensity=intensity,
            num_inference_steps=20,
            guidance_scale=9.0,
            controlnet_conditioning_scale=0.9,
            strength=0.4,
            seed=seed,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False
    elapsed = time.time() - t0

    out = output_dir / f"{Path(image_path).stem}_{procedure}"
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / "input.png"), result["input"])
    cv2.imwrite(str(out / "output.png"), result["output"])
    cv2.imwrite(str(out / "output_raw.png"), result["output_raw"])
    cv2.imwrite(str(out / "conditioning.png"), result["conditioning"])
    cv2.imwrite(str(out / "mask.png"), (result["mask"] * 255).astype(np.uint8))

    # Before/after side-by-side
    before_after = np.hstack([result["input"], result["output"]])
    cv2.imwrite(str(out / "before_after.png"), before_after)

    # Full composite: input | conditioning | mask | output
    mask_vis = (mask_to_3channel(result["mask"]) * 255).astype(np.uint8)
    composite = np.hstack([result["input"], result["conditioning"], mask_vis, result["output"]])
    cv2.imwrite(str(out / "composite.png"), composite)

    print(f"  {procedure} ({intensity:.0f}%): {elapsed:.1f}s -> {out}/")
    return True


def run_all(
    image_path: str | None = None,
    output_dir: str = "scripts/final_output/inference",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading SD1.5 img2img pipeline...")
    pipe = LandmarkDiffPipeline(mode="img2img")
    pipe.load()
    print()

    procedures = [
        "rhinoplasty",
        "blepharoplasty",
        "rhytidectomy",
        "orthognathic",
        "brow_lift",
        "mentoplasty",
    ]

    if image_path:
        images = [image_path]
    else:
        ffhq = Path("data/ffhq_samples")
        images = sorted(str(f) for f in ffhq.glob("*.png"))[:3]

    for img_path in images:
        name = Path(img_path).stem
        print(f"Processing {name}...")
        for proc in procedures:
            run_single(pipe, img_path, proc, intensity=60.0, output_dir=out)
        print()

    # Build a master comparison grid
    _build_master_grid(out, images, procedures)
    print(f"All inference results in {out}/")


def _build_master_grid(out: Path, images: list[str], procedures: list[str]) -> None:
    rows = []
    for img_path in images:
        name = Path(img_path).stem
        panels = []

        # Original
        orig = cv2.imread(img_path)
        if orig is None:
            continue
        orig = cv2.resize(orig, (200, 200))
        cv2.putText(orig, "Original", (5, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        panels.append(orig)

        for proc in procedures:
            result_path = out / f"{name}_{proc}" / "output.png"
            if result_path.exists():
                img = cv2.imread(str(result_path))
                if img is not None:
                    img = cv2.resize(img, (200, 200))
                    cv2.putText(
                        img, proc[:8], (5, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                    )
                    panels.append(img)

        if len(panels) > 1:
            rows.append(np.hstack(panels))

    if rows:
        max_w = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded.append(r)
        grid = np.vstack(padded)
        cv2.imwrite(str(out / "master_grid.png"), grid)
        print(f"Master grid: {out / 'master_grid.png'} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LandmarkDiff inference")
    parser.add_argument(
        "--image", default=None, help="Path to face image (or run on all FFHQ samples)"
    )
    parser.add_argument("--output", default="scripts/final_output/inference")
    args = parser.parse_args()

    run_all(args.image, args.output)
