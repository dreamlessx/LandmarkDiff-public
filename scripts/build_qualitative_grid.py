"""Build qualitative comparison grid from existing predictions.

Creates a figure with one row per procedure:
  Input | Conditioning | LandmarkDiff (Ours) | Ground Truth

Uses pre-computed predictions from paper/predictions/ and test data
from data/hda_splits/test/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _add_label(
    img: np.ndarray,
    text: str,
    font_scale: float = 0.5,
    bg_alpha: float = 0.65,
) -> np.ndarray:
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = 5, th + 5
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 2, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    return img


def build_grid(
    test_dir: Path,
    pred_dir: Path,
    output_path: Path,
    panel_size: int = 256,
    samples: dict[str, str] | None = None,
) -> None:
    """Build a qualitative comparison grid.

    Args:
        test_dir: Directory with *_input.png, *_conditioning.png, *_target.png
        pred_dir: Directory with *_predicted.png
        output_path: Where to save the grid
        panel_size: Size of each panel
        samples: Optional dict of procedure -> prefix to use specific samples
    """
    if samples is None:
        # Pick one representative sample per procedure (manually chosen for quality)
        samples = {
            "rhinoplasty": "rhinoplasty_Nose_30",
            "blepharoplasty": "blepharoplasty_Eyebrow_54",
            "rhytidectomy": "rhytidectomy_Facelift_24",
            "orthognathic": "orthognathic_FacialBones_27",
        }

    col_labels = ["Input", "Conditioning", "LandmarkDiff (Ours)", "Ground Truth"]
    n_rows = len(samples)
    n_cols = len(col_labels)
    header_h = 30
    row_label_w = 100

    grid_h = header_h + n_rows * panel_size
    grid_w = row_label_w + n_cols * panel_size
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    # Column headers
    font = cv2.FONT_HERSHEY_SIMPLEX
    for j, label in enumerate(col_labels):
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        x = row_label_w + j * panel_size + (panel_size - tw) // 2
        y = header_h - 8
        cv2.putText(grid, label, (x, y), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    for i, (procedure, prefix) in enumerate(samples.items()):
        # Row label
        proc_label = procedure.capitalize()
        (tw, th), _ = cv2.getTextSize(proc_label, font, 0.4, 1)
        x = (row_label_w - tw) // 2
        y = header_h + i * panel_size + panel_size // 2 + th // 2
        cv2.putText(grid, proc_label, (x, y), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # Load images
        input_path = test_dir / f"{prefix}_input.png"
        cond_path = test_dir / f"{prefix}_conditioning.png"
        target_path = test_dir / f"{prefix}_target.png"
        pred_path = pred_dir / f"{prefix}_predicted.png"

        imgs = []
        for p in [input_path, cond_path, pred_path, target_path]:
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    img = cv2.resize(img, (panel_size, panel_size))
                    imgs.append(img)
                else:
                    imgs.append(np.ones((panel_size, panel_size, 3), dtype=np.uint8) * 200)
            else:
                print(f"  Missing: {p}")
                imgs.append(np.ones((panel_size, panel_size, 3), dtype=np.uint8) * 200)

        for j, img in enumerate(imgs):
            r0 = header_h + i * panel_size
            c0 = row_label_w + j * panel_size
            grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = img

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    print(f"Grid saved: {output_path} ({grid.shape[1]}x{grid.shape[0]})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build qualitative comparison grid")
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=Path("data/hda_splits/test"),
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        default=Path("paper/predictions"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/fig_qualitative.png"),
    )
    parser.add_argument("--panel_size", type=int, default=256)
    args = parser.parse_args()

    build_grid(args.test_dir, args.pred_dir, args.output, args.panel_size)


if __name__ == "__main__":
    main()
