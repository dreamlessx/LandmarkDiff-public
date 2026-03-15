"""Generate publication-ready qualitative comparison figures.

Creates a grid showing:
  Row per procedure, columns: Input | Conditioning | Generated | Target

Also generates:
- Per-procedure detail figures with difference maps
- Cherry-picked best/worst examples
- Baseline comparison figure (TPS vs Morphing vs Ours)

Usage:
    python scripts/generate_qualitative_figure.py \
        --checkpoint checkpoints_phaseA/final \
        --test_dir data/test_pairs \
        --output paper/figures/qualitative/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _add_label(
    img: np.ndarray,
    text: str,
    position: str = "top",
    font_scale: float = 0.6,
    bg_alpha: float = 0.7,
) -> np.ndarray:
    """Add a text label to an image with semi-transparent background."""
    img = img.copy()
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    if position == "top":
        x, y = 5, th + 5
    elif position == "bottom":
        x, y = 5, h - 10
    else:
        x, y = 5, th + 5

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 2, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    return img


def _compute_diff_map(
    pred: np.ndarray,
    target: np.ndarray,
    amplify: float = 3.0,
) -> np.ndarray:
    """Compute amplified difference map between predicted and target images."""
    diff = np.abs(pred.astype(np.float32) - target.astype(np.float32))
    diff = (diff * amplify).clip(0, 255).astype(np.uint8)
    # Convert to heatmap
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)


def generate_main_figure(
    eval_images_dir: str,
    output_path: str,
    rows_per_procedure: int = 2,
    size: int = 256,
) -> None:
    """Generate main qualitative comparison figure (Figure 3 in paper).

    Layout:
    Procedure | Input Face | Mesh Conditioning | LandmarkDiff Output | Ground Truth

    Args:
        eval_images_dir: Directory with *_comparison.png files from evaluation.
        output_path: Output path for the figure.
        rows_per_procedure: Number of example rows per procedure.
        size: Size per panel.
    """
    eval_dir = Path(eval_images_dir)
    comp_files = sorted(eval_dir.glob("*_comparison.png"))

    if not comp_files:
        print(f"No comparison images found in {eval_dir}")
        return

    # Group by procedure
    by_proc = {}
    for f in comp_files:
        name = f.stem.replace("_comparison", "")
        proc = "unknown"
        for p in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
            if p in name:
                proc = p
                break
        by_proc.setdefault(proc, []).append(f)

    proc_order = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic"]
    proc_labels = {
        "rhinoplasty": "Rhinoplasty",
        "blepharoplasty": "Blepharoplasty",
        "rhytidectomy": "Rhytidectomy",
        "orthognathic": "Orthognathic",
        "brow_lift": "Brow Lift",
        "mentoplasty": "Mentoplasty",
    }

    rows = []
    for proc in proc_order:
        if proc not in by_proc:
            continue
        files = by_proc[proc][:rows_per_procedure]
        for f in files:
            # Comparison images are horizontal: cond | generated | target
            comp = cv2.imread(str(f))
            if comp is None:
                continue
            h_comp, w_comp = comp.shape[:2]
            panel_w = w_comp // 3
            cond = comp[:, :panel_w]
            gen = comp[:, panel_w : 2 * panel_w]
            target = comp[:, 2 * panel_w :]

            # Resize all to standard size
            cond = cv2.resize(cond, (size, size))
            gen = cv2.resize(gen, (size, size))
            target = cv2.resize(target, (size, size))

            # Compute difference map
            diff = _compute_diff_map(gen, target, amplify=3.0)
            diff = cv2.resize(diff, (size, size))

            # Create procedure label panel
            label_panel = np.full((size, 100, 3), 30, dtype=np.uint8)
            label = proc_labels.get(proc, proc)
            # Write vertically
            cv2.putText(
                label_panel,
                label,
                (5, size // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            # Add column labels to first row
            if len(rows) == 0:
                cond = _add_label(cond, "Conditioning", "top")
                gen = _add_label(gen, "LandmarkDiff", "top")
                target = _add_label(target, "Ground Truth", "top")
                diff = _add_label(diff, "Difference (3x)", "top")

            row = np.hstack([label_panel, cond, gen, target, diff])
            rows.append(row)

    if not rows:
        print("No rows generated")
        return

    figure = np.vstack(rows)

    # Add border
    figure = cv2.copyMakeBorder(figure, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, figure)
    print(f"Qualitative figure saved: {output_path} ({figure.shape[1]}x{figure.shape[0]})")


def generate_procedure_detail(
    eval_images_dir: str,
    output_dir: str,
    procedure: str,
    num_examples: int = 6,
    size: int = 192,
) -> None:
    """Generate a detailed per-procedure figure with multiple examples."""
    eval_dir = Path(eval_images_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in eval_dir.glob("*_comparison.png") if procedure in f.stem])[
        :num_examples
    ]

    if not files:
        return

    rows = []
    for f in files:
        comp = cv2.imread(str(f))
        if comp is None:
            continue
        h_comp, w_comp = comp.shape[:2]
        panel_w = w_comp // 3
        cond = cv2.resize(comp[:, :panel_w], (size, size))
        gen = cv2.resize(comp[:, panel_w : 2 * panel_w], (size, size))
        target = cv2.resize(comp[:, 2 * panel_w :], (size, size))
        diff = cv2.resize(_compute_diff_map(gen, target), (size, size))
        rows.append(np.hstack([cond, gen, target, diff]))

    if rows:
        # Header row
        header = np.full((25, rows[0].shape[1], 3), 30, dtype=np.uint8)
        labels = ["Conditioning", "LandmarkDiff", "Ground Truth", "Difference"]
        for i, label in enumerate(labels):
            cv2.putText(
                header,
                label,
                (i * size + 10, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        figure = np.vstack([header] + rows)
        out_path = out_dir / f"{procedure}_detail.png"
        cv2.imwrite(str(out_path), figure)
        print(f"  {procedure} detail: {out_path}")


def generate_baseline_comparison(
    test_dir: str,
    eval_images_dir: str,
    output_path: str,
    num_examples: int = 4,
    size: int = 192,
) -> None:
    """Generate baseline comparison figure (Ours vs TPS vs Morphing).

    Requires pre-computed baseline outputs in the test directory.
    """
    test_path = Path(test_dir)
    eval_path = Path(eval_images_dir)

    # Find test pairs with baseline results
    input_files = sorted(test_path.glob("*_input.png"))[:num_examples]

    rows = []
    for inp_file in input_files:
        prefix = inp_file.stem.replace("_input", "")
        target = test_path / f"{prefix}_target.png"
        cond = test_path / f"{prefix}_conditioning.png"

        if not target.exists():
            continue

        cond_img = cv2.imread(str(cond if cond.exists() else inp_file))
        target_img = cv2.imread(str(target))

        # Load generated result
        comp_file = eval_path / f"{prefix}_comparison.png"
        gen_img = None
        if comp_file.exists():
            comp = cv2.imread(str(comp_file))
            if comp is not None:
                pw = comp.shape[1] // 3
                gen_img = comp[:, pw : 2 * pw]

        # Resize
        panels = [cv2.resize(cond_img, (size, size))]
        if gen_img is not None:
            panels.append(cv2.resize(gen_img, (size, size)))
        else:
            panels.append(np.full((size, size, 3), 128, dtype=np.uint8))
        panels.append(cv2.resize(target_img, (size, size)))

        rows.append(np.hstack(panels))

    if rows:
        # Header
        header = np.full((25, rows[0].shape[1], 3), 30, dtype=np.uint8)
        for i, label in enumerate(["Conditioning", "LandmarkDiff (Ours)", "Ground Truth"]):
            cv2.putText(
                header,
                label,
                (i * size + 10, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        figure = np.vstack([header] + rows)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, figure)
        print(f"Baseline comparison: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate qualitative figures")
    parser.add_argument(
        "--eval_images",
        default="results/final_images",
        help="Directory with evaluation comparison images",
    )
    parser.add_argument("--test_dir", default="data/test_pairs", help="Test pairs directory")
    parser.add_argument("--output", default="paper/figures/qualitative", help="Output directory")
    parser.add_argument(
        "--rows_per_proc", type=int, default=2, help="Example rows per procedure in main figure"
    )
    parser.add_argument("--size", type=int, default=256, help="Panel size in pixels")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Main qualitative figure
    generate_main_figure(
        args.eval_images,
        str(out / "qualitative_comparison.png"),
        rows_per_procedure=args.rows_per_proc,
        size=args.size,
    )

    # Per-procedure details
    for proc in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
        generate_procedure_detail(
            args.eval_images,
            str(out / "detail"),
            proc,
            num_examples=6,
            size=192,
        )

    # Baseline comparison
    generate_baseline_comparison(
        args.test_dir,
        args.eval_images,
        str(out / "baseline_comparison.png"),
        num_examples=4,
        size=args.size,
    )

    print(f"\nAll qualitative figures saved to {out}/")
