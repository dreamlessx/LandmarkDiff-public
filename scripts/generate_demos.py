#!/usr/bin/env python3
"""Generate showcase demo images for LandmarkDiff.

Creates publication-quality before/after grids for all 4 procedures
at multiple intensities, with full neural net post-processing pipeline.

Usage:
    python scripts/generate_demos.py --num-faces 8 --output demos/showcase
    python scripts/generate_demos.py --input data/celeba_hq_extracted --num-faces 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.conditioning import generate_conditioning
from landmarkdiff.evaluation import classify_fitzpatrick_ita
from landmarkdiff.inference import mask_composite
from landmarkdiff.landmarks import extract_landmarks, render_landmark_image, visualize_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.postprocess import full_postprocess
from landmarkdiff.synthetic.tps_warp import warp_image_tps

PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
INTENSITIES = [40, 65, 90]


def process_single(
    image_bgr: np.ndarray,
    procedure: str,
    intensity: float,
    use_neural: bool = True,
) -> dict | None:
    """Run full pipeline on a single image.

    Returns dict with all intermediate and final images, or None if face detection fails.
    """
    img = cv2.resize(image_bgr, (512, 512))
    face = extract_landmarks(img)
    if face is None:
        return None

    manip = apply_procedure_preset(face, procedure, intensity, image_size=512)
    mask = generate_surgical_mask(face, procedure, 512, 512)
    landmark_img = render_landmark_image(manip, 512, 512)
    _, canny, wireframe = generate_conditioning(manip, 512, 512)

    # TPS warp + composite
    warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
    composited = mask_composite(warped, img, mask)

    # Neural net post-processing
    if use_neural:
        try:
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
                identity_threshold=0.6,
            )
            enhanced = pp["image"]
            identity = pp.get("identity_check", {})
        except Exception as e:
            print(f"  Neural postprocess failed: {e}")
            enhanced = composited
            identity = {}
    else:
        enhanced = composited
        identity = {}

    # Skin tone
    try:
        fitz = classify_fitzpatrick_ita(img)
    except Exception:
        fitz = "?"

    return {
        "original": img,
        "landmarks": visualize_landmarks(img, face, radius=2),
        "conditioning": landmark_img,
        "canny": canny,
        "wireframe": wireframe,
        "mask": (mask * 255).astype(np.uint8),
        "warped": warped,
        "composited_raw": composited,
        "composited_neural": enhanced,
        "identity": identity,
        "fitzpatrick": fitz,
        "procedure": procedure,
        "intensity": intensity,
    }


def make_comparison_grid(results: list[dict], title: str = "") -> np.ndarray:
    """Create a before/after comparison grid.

    Layout: each row = one face, columns = [Original, Raw TPS, Neural Enhanced]
    """
    rows = []
    cell_size = 256

    for r in results:
        orig = cv2.resize(r["original"], (cell_size, cell_size))
        raw = cv2.resize(r["composited_raw"], (cell_size, cell_size))
        neural = cv2.resize(r["composited_neural"], (cell_size, cell_size))
        row = np.hstack([orig, raw, neural])
        rows.append(row)

    grid = np.vstack(rows)

    # Add header
    header_h = 40
    header = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8)
    labels = ["Original", "TPS Warp", "Neural Enhanced"]
    for i, label in enumerate(labels):
        x = i * cell_size + cell_size // 2 - len(label) * 5
        cv2.putText(header, label, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if title:
        title_bar = np.zeros((30, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
        return np.vstack([title_bar, header, grid])

    return np.vstack([header, grid])


def make_procedure_grid(face_img: np.ndarray, results_by_proc: dict) -> np.ndarray | None:
    """Create a 4-procedure comparison for a single face.

    Layout: columns = [Original, Rhinoplasty, Blepharoplasty, Rhytidectomy, Orthognathic]
    """
    cell_size = 256
    orig = cv2.resize(face_img, (cell_size, cell_size))

    cols = [orig]
    for proc in PROCEDURES:
        if proc in results_by_proc and results_by_proc[proc] is not None:
            enhanced = cv2.resize(
                results_by_proc[proc]["composited_neural"], (cell_size, cell_size)
            )
            cols.append(enhanced)
        else:
            cols.append(np.zeros_like(orig))

    row = np.hstack(cols)

    # Header
    header_h = 35
    header = np.zeros((header_h, row.shape[1], 3), dtype=np.uint8)
    labels = ["Original"] + [p.capitalize() for p in PROCEDURES]
    for i, label in enumerate(labels):
        x = i * cell_size + 10
        cv2.putText(header, label, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return np.vstack([header, row])


def make_intensity_strip(
    face_img: np.ndarray, procedure: str, use_neural: bool = True
) -> np.ndarray | None:
    """Create intensity sweep strip for one face and procedure.

    Layout: [Original, 20%, 40%, 60%, 80%, 100%]
    """
    cell_size = 200
    intensities = [0, 20, 40, 60, 80, 100]
    cols = []

    for inten in intensities:
        if inten == 0:
            cols.append(cv2.resize(face_img, (cell_size, cell_size)))
            continue
        r = process_single(face_img, procedure, inten, use_neural=use_neural)
        if r is None:
            cols.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))
        else:
            cols.append(cv2.resize(r["composited_neural"], (cell_size, cell_size)))

    strip = np.hstack(cols)

    # Header
    header_h = 30
    header = np.zeros((header_h, strip.shape[1], 3), dtype=np.uint8)
    for i, inten in enumerate(intensities):
        label = "Original" if inten == 0 else f"{inten}%"
        x = i * cell_size + cell_size // 2 - len(label) * 5
        cv2.putText(header, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return np.vstack([header, strip])


def make_pipeline_breakdown(result: dict) -> np.ndarray:
    """Show full pipeline breakdown for one result.

    Layout: [Original, Landmarks, Conditioning, Mask, TPS Warp, Raw Composite, Neural Enhanced]
    """
    cell_size = 180
    items = [
        ("Original", result["original"]),
        ("Landmarks", result["landmarks"]),
        ("Conditioning", result["conditioning"]),
        (
            "Mask",
            cv2.cvtColor(result["mask"], cv2.COLOR_GRAY2BGR)
            if result["mask"].ndim == 2
            else result["mask"],
        ),
        ("TPS Warp", result["warped"]),
        ("Raw Comp.", result["composited_raw"]),
        ("Neural", result["composited_neural"]),
    ]

    cols = []
    for _, img in items:
        cols.append(cv2.resize(img, (cell_size, cell_size)))
    strip = np.hstack(cols)

    header_h = 30
    header = np.zeros((header_h, strip.shape[1], 3), dtype=np.uint8)
    for i, (label, _) in enumerate(items):
        x = i * cell_size + 5
        cv2.putText(header, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return np.vstack([header, strip])


def load_faces(input_dir: str, num_faces: int, seed: int = 42) -> list[np.ndarray]:
    """Load and validate face images from a directory."""
    rng = np.random.default_rng(seed)
    image_dir = Path(input_dir)
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    all_files = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in extensions)

    if not all_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    # Shuffle and pick candidates — we may need more than num_faces if some fail detection
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

    print(f"Loaded {len(faces)} faces with valid landmarks from {image_dir.name}/")
    return faces


def main():
    parser = argparse.ArgumentParser(description="Generate LandmarkDiff demo showcase")
    parser.add_argument(
        "--input", default="data/celeba_hq_extracted", help="Directory of face images"
    )
    parser.add_argument(
        "--output", default="demos/showcase", help="Output directory for demo images"
    )
    parser.add_argument("--num-faces", type=int, default=8, help="Number of faces to use")
    parser.add_argument(
        "--no-neural", action="store_true", help="Skip neural post-processing (faster)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_neural = not args.no_neural

    print(f"Loading faces from {args.input}...")
    faces = load_faces(args.input, args.num_faces, args.seed)
    if not faces:
        print("No valid faces found. Exiting.")
        return

    # 1. Per-face, all-procedures grid
    print("\n--- Generating procedure comparison grids ---")
    proc_rows = []
    for i, face in enumerate(faces):
        print(f"  Face {i + 1}/{len(faces)}...")
        results_by_proc = {}
        for proc in PROCEDURES:
            r = process_single(face, proc, 65.0, use_neural=use_neural)
            results_by_proc[proc] = r
        grid = make_procedure_grid(face, results_by_proc)
        if grid is not None:
            proc_rows.append(grid)
            cv2.imwrite(str(output_dir / f"procedures_face_{i:02d}.png"), grid)

    if proc_rows:
        full_grid = np.vstack(proc_rows)
        cv2.imwrite(str(output_dir / "all_procedures_grid.png"), full_grid)
        print(f"  Saved all_procedures_grid.png ({full_grid.shape[1]}x{full_grid.shape[0]})")

    # 2. Before/after comparison grids per procedure
    print("\n--- Generating before/after grids ---")
    for proc in PROCEDURES:
        results = []
        for face in faces[:6]:
            r = process_single(face, proc, 65.0, use_neural=use_neural)
            if r is not None:
                results.append(r)
        if results:
            grid = make_comparison_grid(results, title=f"{proc.capitalize()} (65% intensity)")
            cv2.imwrite(str(output_dir / f"comparison_{proc}.png"), grid)
            print(f"  Saved comparison_{proc}.png")

    # 3. Intensity sweep strips
    print("\n--- Generating intensity sweeps ---")
    for proc in PROCEDURES:
        strip = make_intensity_strip(faces[0], proc, use_neural=use_neural)
        if strip is not None:
            cv2.imwrite(str(output_dir / f"sweep_{proc}.png"), strip)
            print(f"  Saved sweep_{proc}.png")

    # 4. Pipeline breakdown
    print("\n--- Generating pipeline breakdowns ---")
    for i, face in enumerate(faces[:3]):
        r = process_single(face, "rhinoplasty", 65.0, use_neural=use_neural)
        if r is not None:
            breakdown = make_pipeline_breakdown(r)
            cv2.imwrite(str(output_dir / f"pipeline_{i:02d}.png"), breakdown)
            print(f"  Saved pipeline_{i:02d}.png")

    # 5. High-res before/after pairs (individual files)
    print("\n--- Generating high-res before/after pairs ---")
    pair_dir = output_dir / "pairs"
    pair_dir.mkdir(exist_ok=True)
    for i, face in enumerate(faces[:4]):
        for proc in PROCEDURES:
            r = process_single(face, proc, 65.0, use_neural=use_neural)
            if r is not None:
                before_after = np.hstack([r["original"], r["composited_neural"]])
                cv2.imwrite(str(pair_dir / f"face{i}_{proc}_before_after.png"), before_after)
    print(f"  Saved {len(list(pair_dir.glob('*.png')))} pair images")

    # 6. Fitzpatrick diversity report
    print("\n--- Fitzpatrick diversity ---")
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
