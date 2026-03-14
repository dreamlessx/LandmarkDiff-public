"""Generate qualitative comparison figure for paper.

Creates a grid showing input → conditioning → baselines → LandmarkDiff → target
for representative examples from each procedure.

Usage:
    python scripts/generate_comparison_figure.py \
        --predictions paper/predictions/ \
        --test_dir data/hda_splits/test \
        --output paper/fig_comparison.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def add_text_label(img: np.ndarray, text: str, font_scale: float = 0.5) -> np.ndarray:
    """Add a white text label at the bottom of the image."""
    h, w = img.shape[:2]
    labeled = img.copy()
    # Black bar at bottom
    bar_h = int(20 * font_scale * 2)
    labeled[h - bar_h :, :] = 0
    # White text
    cv2.putText(
        labeled,
        text,
        (5, h - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def make_comparison_row(
    prefix: str,
    test_dir: Path,
    predictions_dir: Path,
    sd_outputs_dir: Path | None,
    size: int = 256,
) -> np.ndarray | None:
    """Create a single comparison row for a test pair."""
    input_path = test_dir / f"{prefix}_input.png"
    target_path = test_dir / f"{prefix}_target.png"
    conditioning_path = test_dir / f"{prefix}_conditioning.png"
    pred_path = predictions_dir / f"{prefix}_predicted.png"
    sd_path = sd_outputs_dir / f"{prefix}_sd_img2img.png" if sd_outputs_dir else None

    # Load images
    input_img = cv2.imread(str(input_path))
    target_img = cv2.imread(str(target_path))

    if input_img is None or target_img is None:
        return None

    # Resize all to consistent size
    input_img = cv2.resize(input_img, (size, size))
    target_img = cv2.resize(target_img, (size, size))

    row_images = [add_text_label(input_img, "Input")]

    # Conditioning (if exists)
    if conditioning_path.exists():
        cond_img = cv2.imread(str(conditioning_path))
        if cond_img is not None:
            cond_img = cv2.resize(cond_img, (size, size))
            row_images.append(add_text_label(cond_img, "Conditioning"))

    # TPS baseline (compute on the fly)
    try:
        from landmarkdiff.landmarks import extract_landmarks
        from landmarkdiff.synthetic.tps_warp import warp_image_tps

        input_lm = extract_landmarks(input_img)
        target_lm = extract_landmarks(target_img)
        if input_lm is not None and target_lm is not None:
            tps_warped = warp_image_tps(input_img, input_lm.pixel_coords, target_lm.pixel_coords)
            row_images.append(add_text_label(tps_warped, "TPS-only"))
    except Exception:
        # Add placeholder
        placeholder = np.ones((size, size, 3), dtype=np.uint8) * 128
        row_images.append(add_text_label(placeholder, "TPS-only"))

    # SD Img2Img baseline
    if sd_path and sd_path.exists():
        sd_img = cv2.imread(str(sd_path))
        if sd_img is not None:
            sd_img = cv2.resize(sd_img, (size, size))
            row_images.append(add_text_label(sd_img, "SD1.5 Img2Img"))

    # LandmarkDiff prediction
    if pred_path.exists():
        pred_img = cv2.imread(str(pred_path))
        if pred_img is not None:
            pred_img = cv2.resize(pred_img, (size, size))
            row_images.append(add_text_label(pred_img, "LandmarkDiff"))
    else:
        placeholder = np.ones((size, size, 3), dtype=np.uint8) * 128
        row_images.append(add_text_label(placeholder, "LandmarkDiff"))

    # Ground truth target
    row_images.append(add_text_label(target_img, "Target (GT)"))

    # Concatenate horizontally
    return np.hstack(row_images)


def select_representative_pairs(test_dir: Path, n_per_proc: int = 2) -> dict:
    """Select representative test pairs for each procedure."""
    pairs_by_proc = {}
    for f in sorted(test_dir.glob("*_input.png")):
        prefix = f.stem.replace("_input", "")
        procedure = prefix.split("_")[0]
        target = test_dir / f"{prefix}_target.png"
        if target.exists():
            pairs_by_proc.setdefault(procedure, []).append(prefix)

    selected = {}
    for proc, prefixes in sorted(pairs_by_proc.items()):
        # Select evenly spaced examples
        indices = np.linspace(0, len(prefixes) - 1, min(n_per_proc, len(prefixes)), dtype=int)
        selected[proc] = [prefixes[i] for i in indices]

    return selected


def main():
    parser = argparse.ArgumentParser(description="Generate comparison figure")
    parser.add_argument("--test_dir", type=Path, default=ROOT / "data" / "hda_splits" / "test")
    parser.add_argument("--predictions", type=Path, default=ROOT / "paper" / "predictions")
    parser.add_argument("--sd_outputs", type=Path, default=ROOT / "paper" / "sd_img2img_outputs")
    parser.add_argument("--output", type=Path, default=ROOT / "paper" / "fig_comparison.png")
    parser.add_argument("--n_per_proc", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    print("Selecting representative test pairs...")
    selected = select_representative_pairs(args.test_dir, args.n_per_proc)

    for proc, prefixes in selected.items():
        print(f"  {proc}: {prefixes}")

    print("\nGenerating comparison rows...")
    rows = []
    proc_labels = []

    for proc, prefixes in sorted(selected.items()):
        for prefix in prefixes:
            row = make_comparison_row(
                prefix,
                args.test_dir,
                args.predictions,
                args.sd_outputs if args.sd_outputs.exists() else None,
                size=args.size,
            )
            if row is not None:
                rows.append(row)
                proc_labels.append(proc)
                print(f"  {proc}/{prefix}: {row.shape}")

    if not rows:
        print("No rows generated — check paths")
        return

    # Make all rows the same width (pad shorter ones)
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    # Add procedure labels on the left
    label_w = 100
    labeled_rows = []
    for row, proc in zip(padded, proc_labels, strict=False):
        label = np.zeros((row.shape[0], label_w, 3), dtype=np.uint8)
        # Rotate text for procedure name
        cv2.putText(
            label,
            proc.capitalize(),
            (5, row.shape[0] // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        labeled_rows.append(np.hstack([label, row]))

    # Stack vertically with thin separator
    separator = np.ones((2, labeled_rows[0].shape[1], 3), dtype=np.uint8) * 128
    final_rows = []
    for i, row in enumerate(labeled_rows):
        if i > 0:
            final_rows.append(separator)
        final_rows.append(row)

    figure = np.vstack(final_rows)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), figure)
    print(f"\nSaved comparison figure: {args.output} ({figure.shape})")


if __name__ == "__main__":
    main()
