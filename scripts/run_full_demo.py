"""Full demo - run inference on multiple faces, build a results grid."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.conditioning import generate_conditioning
from landmarkdiff.inference import LandmarkDiffPipeline
from landmarkdiff.landmarks import extract_landmarks, render_landmark_image
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask, mask_to_3channel


def add_text(img: np.ndarray, text: str, pos: str = "bottom") -> np.ndarray:
    result = img.copy()
    h, w = result.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    if pos == "bottom":
        y = h - 8
        cv2.rectangle(result, (0, h - th - 12), (tw + 10, h), (0, 0, 0), -1)
    else:
        y = th + 8
        cv2.rectangle(result, (0, 0), (tw + 10, th + 12), (0, 0, 0), -1)

    cv2.putText(result, text, (5, y), font, scale, (255, 255, 255), thickness)
    return result


def run(
    input_dir: str = "data/ffhq_samples",
    output_dir: str = "scripts/final_output/results",
    procedures: list[str] | None = None,
    num_images: int = 5,
) -> None:
    """Run full demo inference on multiple faces.

    Args:
        input_dir: Directory containing input face images
        output_dir: Directory to save results
        procedures: List of procedures to run (default: all 4)
        num_images: Number of images to process
    """
    if procedures is None:
        procedures = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ffhq = sorted(Path(input_dir).glob("*.png"))[:num_images]
    if not ffhq:
        ffhq = sorted(Path(input_dir).glob("*.jpg"))[:num_images]

    print("Loading pipeline...")
    pipe = LandmarkDiffPipeline(mode="img2img")
    pipe.load()
    print()

    all_rows = []

    for img_path in ffhq:
        name = img_path.stem
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.resize(image, (512, 512))

        face = extract_landmarks(image)
        if face is None:
            print(f"  {name}: no face detected, skipping")
            continue

        row_panels = [add_text(cv2.resize(image, (200, 200)), f"{name} Original")]

        for proc in procedures:
            t0 = time.time()

            # Pipeline conditioning (pre-diffusion)
            manipulated = apply_procedure_preset(face, proc, intensity=70.0, image_size=512)
            _, _, orig_wf = generate_conditioning(face, 512, 512)
            _, _, manip_wf = generate_conditioning(manipulated, 512, 512)
            mask = generate_surgical_mask(face, proc, 512, 512)
            landmark_img = render_landmark_image(manipulated, 512, 512)

            # Run diffusion
            result = pipe.generate(
                image,
                procedure=proc,
                intensity=70.0,
                num_inference_steps=25,
                guidance_scale=9.0,
                strength=0.55,
                seed=42,
            )
            elapsed = time.time() - t0

            # Save individual outputs
            proc_dir = out / f"{name}_{proc}"
            proc_dir.mkdir(exist_ok=True)

            cv2.imwrite(str(proc_dir / "input.png"), image)
            cv2.imwrite(str(proc_dir / "output.png"), result["output"])
            cv2.imwrite(str(proc_dir / "raw.png"), result["output_raw"])
            cv2.imwrite(str(proc_dir / "conditioning.png"), landmark_img)
            cv2.imwrite(str(proc_dir / "mask.png"), (mask * 255).astype(np.uint8))

            # Wireframe diff overlay
            wf_diff = cv2.absdiff(orig_wf, manip_wf)
            diff_color = np.zeros((512, 512, 3), dtype=np.uint8)
            diff_color[:, :, 2] = wf_diff  # red = displacement
            diff_overlay = cv2.addWeighted(image, 0.6, diff_color, 0.8, 0)
            mask_3ch = mask_to_3channel(mask)
            diff_overlay = (
                diff_overlay.astype(np.float32) * mask_3ch
                + image.astype(np.float32) * (1 - mask_3ch * 0.5)
            ).astype(np.uint8)
            cv2.imwrite(str(proc_dir / "diff_overlay.png"), diff_overlay)

            # Before/after
            ba = np.hstack([image, result["output"]])
            cv2.imwrite(str(proc_dir / "before_after.png"), ba)

            # Full strip: input | wireframe diff | conditioning | mask | output
            strip = np.hstack(
                [
                    image,
                    diff_overlay,
                    cv2.cvtColor(manip_wf, cv2.COLOR_GRAY2BGR),
                    (mask_3ch * 255).astype(np.uint8),
                    result["output"],
                ]
            )
            cv2.imwrite(str(proc_dir / "full_strip.png"), strip)

            # Add to grid
            panel = cv2.resize(result["output"], (200, 200))
            panel = add_text(panel, f"{proc[:8]} {elapsed:.1f}s")
            row_panels.append(panel)

            print(f"  {name} {proc}: {elapsed:.1f}s")

        all_rows.append(np.hstack(row_panels))

    if all_rows:
        max_w = max(r.shape[1] for r in all_rows)
        padded = []
        for r in all_rows:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded.append(r)
        grid = np.vstack(padded)
        cv2.imwrite(str(out / "master_results.png"), grid)
        print(
            f"\nMaster results grid: {out / 'master_results.png'} ({grid.shape[1]}x{grid.shape[0]})"
        )

    print(f"All results in {out}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full demo - run inference on multiple faces.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/ffhq_samples",
        help="Input directory with face images (default: data/ffhq_samples)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/final_output/results",
        help="Output directory for results (default: scripts/final_output/results)",
    )
    parser.add_argument(
        "--procedures",
        type=str,
        nargs="+",
        choices=["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"],
        default=None,
        help="Procedures to apply (default: all 6)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=5,
        help="Number of input images to process (default: 5)",
    )

    args = parser.parse_args()
    run(
        input_dir=args.input,
        output_dir=args.output,
        procedures=args.procedures,
        num_images=args.num_images,
    )
