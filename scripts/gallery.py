"""Generate visual result galleries (grid composites per procedure)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    scale = target_h / img.shape[0]
    new_w = int(img.shape[1] * scale)
    resized = cv2.resize(img, (new_w, target_h))
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return resized


def add_label(img: np.ndarray, text: str, font_scale: float = 0.5) -> np.ndarray:
    result = img.copy()
    h, w = result.shape[:2]
    cv2.rectangle(result, (0, h - 25), (w, h), (0, 0, 0), -1)
    cv2.putText(result, text, (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    return result


def make_procedure_comparison(face_dir: str, output_path: str) -> None:
    """Create a 6-procedure comparison grid from demo outputs."""
    procedures = [
        "rhinoplasty",
        "blepharoplasty",
        "rhytidectomy",
        "orthognathic",
        "brow_lift",
        "mentoplasty",
    ]
    base = Path(face_dir)
    rows = []

    for proc in procedures:
        d = base / f"demo_001_{proc}"
        if not d.exists():
            continue

        panels = []
        files = [
            ("Input", "01_landmarks_on_face.png"),
            ("Wireframe", "03_wireframe.png"),
            ("Manipulated", "06_manipulated_wireframe.png"),
            ("Mask", "08_surgical_mask.png"),
            ("Overlay", "09_mask_overlay.png"),
        ]

        for label, fname in files:
            fpath = d / fname
            if fpath.exists():
                img = cv2.imread(str(fpath))
                if img is not None:
                    img = resize_to_height(img, 200)
                    img = add_label(img, f"{proc[:6]} | {label}", 0.35)
                    panels.append(img)

        if panels:
            # Pad all panels to same width
            max_w = max(p.shape[1] for p in panels)
            padded = []
            for p in panels:
                if p.shape[1] < max_w:
                    pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
                    p = np.hstack([p, pad])
                padded.append(p)
            rows.append(np.hstack(padded))

    if rows:
        # Pad rows to same width
        max_w = max(r.shape[1] for r in rows)
        padded_rows = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded_rows.append(r)

        grid = np.vstack(padded_rows)
        cv2.imwrite(output_path, grid)
        print(f"Procedure comparison: {output_path} ({grid.shape[1]}x{grid.shape[0]})")


def make_training_pair_gallery(pair_dir: str, output_path: str, max_pairs: int = 10) -> None:
    """Create a gallery of synthetic training pairs."""
    d = Path(pair_dir)
    rows = []

    for i in range(max_pairs):
        prefix = f"{i:06d}"
        files = {
            "input": d / f"{prefix}_input.png",
            "target": d / f"{prefix}_target.png",
            "conditioning": d / f"{prefix}_conditioning.png",
            "canny": d / f"{prefix}_canny.png",
            "mask": d / f"{prefix}_mask.png",
        }

        if not files["input"].exists():
            break

        panels = []
        for label, fpath in files.items():
            if fpath.exists():
                img = cv2.imread(str(fpath))
                if img is not None:
                    img = resize_to_height(img, 160)
                    img = add_label(img, label, 0.35)
                    panels.append(img)

        if panels:
            max_w = max(p.shape[1] for p in panels)
            padded = []
            for p in panels:
                if p.shape[1] < max_w:
                    pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
                    p = np.hstack([p, pad])
                padded.append(p)
            rows.append(np.hstack(padded))

    if rows:
        max_w = max(r.shape[1] for r in rows)
        padded_rows = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded_rows.append(r)

        grid = np.vstack(padded_rows)
        cv2.imwrite(output_path, grid)
        print(f"Training pair gallery: {output_path} ({grid.shape[1]}x{grid.shape[0]})")


def make_intensity_sweep(
    image_path: str,
    procedure: str,
    output_path: str,
    intensities: list[float] | None = None,
) -> None:
    """Show effect of intensity parameter on a single procedure."""
    from landmarkdiff.conditioning import generate_conditioning
    from landmarkdiff.landmarks import extract_landmarks
    from landmarkdiff.manipulation import apply_procedure_preset
    from landmarkdiff.masking import generate_surgical_mask

    if intensities is None:
        intensities = [10, 30, 50, 70, 90]

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load {image_path}")
        return

    image = cv2.resize(image, (512, 512))
    face = extract_landmarks(image)
    if face is None:
        print("No face detected")
        return

    _, _, orig_wf = generate_conditioning(face, 512, 512)

    panels = [add_label(resize_to_height(image, 256), "Original", 0.4)]
    panels.append(
        add_label(
            resize_to_height(cv2.cvtColor(orig_wf, cv2.COLOR_GRAY2BGR), 256), "Wireframe", 0.4
        )
    )

    for intensity in intensities:
        manip = apply_procedure_preset(face, procedure, intensity, 512)
        _, _, manip_wf = generate_conditioning(manip, 512, 512)
        mask = generate_surgical_mask(face, procedure, 512, 512)

        # Blend wireframe diff with mask overlay
        wf_diff = cv2.absdiff(orig_wf, manip_wf)
        colored = np.zeros((512, 512, 3), dtype=np.uint8)
        colored[:, :, 2] = wf_diff  # red channel = displacement
        colored[:, :, 0] = (mask * 128).astype(np.uint8)  # blue = mask

        panel = resize_to_height(colored, 256)
        panel = add_label(panel, f"I={int(intensity)}", 0.4)
        panels.append(panel)

    row = np.hstack(panels)
    cv2.imwrite(output_path, row)
    print(f"Intensity sweep ({procedure}): {output_path} ({row.shape[1]}x{row.shape[0]})")


def make_before_after_grid(face_dirs: list[str], output_path: str) -> None:
    """Create before/after wireframe comparison grid."""
    rows = []
    for d in face_dirs:
        d = Path(d)
        orig = d / "03_wireframe.png"
        manip = d / "06_manipulated_wireframe.png"
        overlay = d / "09_mask_overlay.png"

        if not orig.exists():
            continue

        panels = []
        for fpath, label in [(orig, "Before"), (manip, "After"), (overlay, "Mask")]:
            if fpath.exists():
                img = cv2.imread(str(fpath))
                if img is not None:
                    img = resize_to_height(img, 200)
                    img = add_label(img, label, 0.4)
                    panels.append(img)

        if panels:
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
        cv2.imwrite(output_path, grid)
        print(f"Before/after grid: {output_path} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate visual result galleries (grid composites per procedure)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("scripts/final_output"),
        help="Input directory containing demo output folders (default: scripts/final_output)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/final_output"),
        help="Output directory for gallery images (default: scripts/final_output)",
    )
    parser.add_argument(
        "--procedure",
        type=str,
        choices=[
            "all",
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
        ],
        default="all",
        help="Specific procedure to generate intensity sweep for (default: all)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images to include in training pair gallery (default: 10)",
    )

    args = parser.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    # 1. Procedure comparison grid
    make_procedure_comparison(str(args.input), str(out / "gallery_procedures.png"))

    # 2. Training pair gallery
    make_training_pair_gallery(
        "data/synthetic_pairs", str(out / "gallery_training_pairs.png"), max_pairs=args.num_images
    )

    # 3. Intensity sweeps
    procedures = (
        ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic"]
        if args.procedure == "all"
        else [args.procedure]
    )
    for proc in procedures:
        make_intensity_sweep(
            "data/ffhq_samples/000001.png",
            proc,
            str(out / f"gallery_intensity_{proc}.png"),
        )

    # 4. Before/after grid from demo_pairs
    if Path("data/demo_pairs").exists():
        make_training_pair_gallery(
            "data/demo_pairs", str(out / "gallery_demo_pairs.png"), max_pairs=4
        )

    print(f"\nAll galleries saved to {out}/")
