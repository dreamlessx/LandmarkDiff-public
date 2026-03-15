#!/usr/bin/env python3
"""Compare different methods/checkpoints side-by-side.

Generates publication-quality comparison grids showing multiple methods
applied to the same input images.

Usage:
    # Compare TPS baseline vs neural post-processed
    python scripts/compare_methods.py \
        --input data/celeba_hq_extracted \
        --output demos/comparison \
        --methods tps neural \
        --num-faces 6

    # Compare different intensities
    python scripts/compare_methods.py \
        --input data/celeba_hq_extracted \
        --output demos/intensity_comparison \
        --procedure rhinoplasty \
        --intensities 20 40 60 80 100 \
        --num-faces 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.evaluation import classify_fitzpatrick_ita, compute_ssim
from landmarkdiff.inference import mask_composite
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps

PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]


def process_tps(img: np.ndarray, procedure: str, intensity: float) -> np.ndarray | None:
    """Process with TPS warp only."""
    face = extract_landmarks(img)
    if face is None:
        return None
    manip = apply_procedure_preset(face, procedure, intensity, image_size=512)
    mask = generate_surgical_mask(face, procedure, 512, 512)
    warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
    return mask_composite(warped, img, mask)


def process_neural(img: np.ndarray, procedure: str, intensity: float) -> np.ndarray | None:
    """Process with TPS + neural post-processing."""
    face = extract_landmarks(img)
    if face is None:
        return None
    manip = apply_procedure_preset(face, procedure, intensity, image_size=512)
    mask = generate_surgical_mask(face, procedure, 512, 512)
    warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
    composited = mask_composite(warped, img, mask)

    try:
        from landmarkdiff.postprocess import full_postprocess

        pp = full_postprocess(
            generated=composited,
            original=img,
            mask=mask,
            restore_mode="codeformer",
            codeformer_fidelity=0.7,
            use_realesrgan=True,
            use_laplacian_blend=True,
            sharpen_strength=0.25,
            verify_identity=True,
        )
        return pp["image"]
    except Exception:
        return composited


def load_faces(input_dir: Path, num_faces: int, seed: int = 42) -> list[np.ndarray]:
    """Load face images with valid landmarks."""
    rng = np.random.default_rng(seed)
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    all_files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in extensions)

    rng.shuffle(all_files)
    faces = []
    for f in all_files:
        if len(faces) >= num_faces:
            break
        img = cv2.imread(str(f))
        if img is None:
            continue
        img_512 = cv2.resize(img, (512, 512))
        if extract_landmarks(img_512) is not None:
            faces.append(img_512)

    return faces


def make_method_comparison(
    faces: list[np.ndarray],
    procedure: str,
    intensity: float,
    methods: list[str],
    cell_size: int = 256,
) -> np.ndarray:
    """Create a methods comparison grid.

    Rows = faces, Columns = [Original, method1, method2, ...]
    """
    method_fns = {
        "tps": process_tps,
        "neural": process_neural,
    }

    all_rows = []
    for face in faces:
        cols = [cv2.resize(face, (cell_size, cell_size))]

        for method in methods:
            fn = method_fns.get(method, process_tps)
            result = fn(face, procedure, intensity)
            if result is not None:
                ssim = compute_ssim(result, face)
                cell = cv2.resize(result, (cell_size, cell_size))
                # Add SSIM label
                cv2.putText(
                    cell,
                    f"SSIM:{ssim:.3f}",
                    (5, cell_size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                )
                cols.append(cell)
            else:
                cols.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))

        all_rows.append(np.hstack(cols))

    grid = np.vstack(all_rows)

    # Header
    header_h = 35
    header = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8)
    labels = ["Original"] + [m.upper() for m in methods]
    for i, label in enumerate(labels):
        x = i * cell_size + 10
        cv2.putText(header, label, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Title
    title_h = 30
    title = np.zeros((title_h, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        title,
        f"{procedure.capitalize()} @ {intensity}% — Method Comparison",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 255),
        1,
    )

    return np.vstack([title, header, grid])


def make_intensity_comparison(
    faces: list[np.ndarray],
    procedure: str,
    intensities: list[float],
    cell_size: int = 200,
) -> np.ndarray:
    """Create an intensity comparison grid.

    Rows = faces, Columns = [Original, intensity1, intensity2, ...]
    """
    all_rows = []
    for face in faces:
        cols = [cv2.resize(face, (cell_size, cell_size))]

        for inten in intensities:
            result = process_tps(face, procedure, inten)
            if result is not None:
                cols.append(cv2.resize(result, (cell_size, cell_size)))
            else:
                cols.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))

        all_rows.append(np.hstack(cols))

    grid = np.vstack(all_rows)

    # Header
    header_h = 30
    header = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8)
    labels = ["Original"] + [f"{int(i)}%" for i in intensities]
    for i, label in enumerate(labels):
        x = i * cell_size + cell_size // 2 - len(label) * 5
        cv2.putText(header, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    title_h = 30
    title = np.zeros((title_h, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        title,
        f"{procedure.capitalize()} — Intensity Sweep",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 255),
        1,
    )

    return np.vstack([title, header, grid])


def make_procedure_comparison(
    faces: list[np.ndarray],
    intensity: float,
    cell_size: int = 200,
) -> np.ndarray:
    """All procedures side by side for each face.

    Rows = faces, Columns = [Original, Rhino, Bleph, Rhytid, Ortho]
    """
    all_rows = []
    for face in faces:
        cols = [cv2.resize(face, (cell_size, cell_size))]
        for proc in PROCEDURES:
            result = process_tps(face, proc, intensity)
            if result is not None:
                cols.append(cv2.resize(result, (cell_size, cell_size)))
            else:
                cols.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))
        all_rows.append(np.hstack(cols))

    grid = np.vstack(all_rows)

    header_h = 30
    header = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8)
    labels = ["Original"] + [p[:5].capitalize() for p in PROCEDURES]
    for i, label in enumerate(labels):
        x = i * cell_size + 10
        cv2.putText(header, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return np.vstack([header, grid])


def main():
    parser = argparse.ArgumentParser(description="Compare methods/checkpoints side-by-side")
    parser.add_argument("--input", required=True, help="Face images directory")
    parser.add_argument("--output", default="demos/comparison", help="Output directory")
    parser.add_argument("--procedure", default="rhinoplasty", choices=PROCEDURES)
    parser.add_argument("--intensity", type=float, default=65.0)
    parser.add_argument(
        "--intensities",
        nargs="+",
        type=float,
        default=None,
        help="Create intensity comparison (overrides --intensity)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["tps", "neural"], help="Methods to compare"
    )
    parser.add_argument(
        "--all-procedures", action="store_true", help="Create all-procedures comparison"
    )
    parser.add_argument("--num-faces", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading faces from {args.input}...")
    faces = load_faces(Path(args.input), args.num_faces, args.seed)
    if not faces:
        print("No valid faces found.")
        return
    print(f"Loaded {len(faces)} faces")

    if args.intensities:
        print(f"\nGenerating intensity comparison: {args.intensities}")
        grid = make_intensity_comparison(faces, args.procedure, args.intensities)
        path = output_dir / f"intensity_{args.procedure}.png"
        cv2.imwrite(str(path), grid)
        print(f"Saved: {path}")

    if args.all_procedures:
        print(f"\nGenerating all-procedures comparison @ {args.intensity}%")
        grid = make_procedure_comparison(faces, args.intensity)
        path = output_dir / "all_procedures.png"
        cv2.imwrite(str(path), grid)
        print(f"Saved: {path}")

    if not args.intensities:
        print(f"\nGenerating method comparison: {args.methods}")
        for proc in [args.procedure] if not args.all_procedures else PROCEDURES:
            grid = make_method_comparison(faces, proc, args.intensity, args.methods)
            path = output_dir / f"methods_{proc}.png"
            cv2.imwrite(str(path), grid)
            print(f"Saved: {path}")

    # Fitzpatrick diversity report
    print("\nFitzpatrick diversity:")
    fitz_counts: dict[str, int] = {}
    for face in faces:
        try:
            fitz = classify_fitzpatrick_ita(face)
            fitz_counts[fitz] = fitz_counts.get(fitz, 0) + 1
        except Exception:
            pass
    for ftype in sorted(fitz_counts):
        print(f"  Type {ftype}: {fitz_counts[ftype]} faces")

    print(f"\nDone! All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
