"""PI demo - presentation-ready figures for all procedures and intensity sweeps."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.conditioning import generate_conditioning
from landmarkdiff.inference import LandmarkDiffPipeline, mask_composite
from landmarkdiff.landmarks import (
    FaceLandmarks,
    extract_landmarks,
    visualize_landmarks,
)
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps

WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
BLACK = (0, 0, 0)
ACCENT = (0, 200, 255)  # yellow-orange in BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_label(img: np.ndarray, text: str, pos: str = "top", scale: float = 0.45) -> np.ndarray:
    result = img.copy()
    h, w = result.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)

    if pos == "top":
        cv2.rectangle(result, (0, 0), (w, th + 14), BLACK, -1)
        cv2.putText(result, text, (6, th + 8), FONT, scale, WHITE, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(result, (0, h - th - 14), (w, h), BLACK, -1)
        cv2.putText(result, text, (6, h - 6), FONT, scale, WHITE, 1, cv2.LINE_AA)

    return result


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def resize_sq(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(ensure_bgr(img), (size, size))


def make_figure_1_pipeline(face: FaceLandmarks, image: np.ndarray, out: Path) -> None:
    """Figure 1: Pipeline architecture - single face, rhinoplasty."""
    s = 256

    # Original with landmarks
    annotated = visualize_landmarks(image, face, radius=2)
    _, canny, wireframe = generate_conditioning(face, 512, 512)

    # Manipulate
    manip = apply_procedure_preset(face, "rhinoplasty", 65.0, 512)
    _, manip_canny, manip_wf = generate_conditioning(manip, 512, 512)
    mask = generate_surgical_mask(face, "rhinoplasty", 512, 512)

    # TPS warp
    tps_result = warp_image_tps(image, face.pixel_coords, manip.pixel_coords)
    tps_blended = mask_composite(tps_result, image, mask)

    panels = [
        put_label(resize_sq(image, s), "1. Input"),
        put_label(resize_sq(annotated, s), "2. 478 Landmarks"),
        put_label(resize_sq(wireframe, s), "3. Wireframe"),
        put_label(resize_sq(manip_wf, s), "4. Manipulated"),
        put_label(resize_sq((mask * 255).astype(np.uint8), s), "5. Surgical Mask"),
        put_label(resize_sq(tps_blended, s), "6. TPS Output"),
    ]

    row = np.hstack(panels)

    # Add title bar
    title_h = 40
    title_bar = np.zeros((title_h, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        title_bar,
        "LandmarkDiff Pipeline: Input -> Landmarks -> Condition -> Manipulate -> Mask -> Generate",
        (10, 28),
        FONT,
        0.5,
        ACCENT,
        1,
        cv2.LINE_AA,
    )
    figure = np.vstack([title_bar, row])
    cv2.imwrite(str(out / "fig1_pipeline.png"), figure)
    print("  Fig 1: Pipeline overview")


def make_figure_2_procedures(
    face: FaceLandmarks,
    image: np.ndarray,
    out: Path,
    pipe: LandmarkDiffPipeline | None = None,
) -> None:
    """Figure 2: All 6 procedures side by side."""
    s = 200
    procedures = [
        "rhinoplasty",
        "blepharoplasty",
        "rhytidectomy",
        "orthognathic",
        "brow_lift",
        "mentoplasty",
    ]
    short_names = [
        "Rhinoplasty",
        "Blepharoplasty",
        "Rhytidectomy",
        "Orthognathic",
        "Brow Lift",
        "Mentoplasty",
    ]

    cols = [put_label(resize_sq(image, s), "Original")]

    for proc, name in zip(procedures, short_names, strict=False):
        manip = apply_procedure_preset(face, proc, 65.0, 512)
        mask = generate_surgical_mask(face, proc, 512, 512)
        tps = warp_image_tps(image, face.pixel_coords, manip.pixel_coords)
        blended = mask_composite(tps, image, mask)

        if pipe and pipe.is_loaded and pipe.mode != "tps":
            result = pipe.generate(image, proc, 65.0, seed=42)
            panel = resize_sq(result["output"], s)
        else:
            panel = resize_sq(blended, s)

        cols.append(put_label(panel, name))

    row = np.hstack(cols)
    cv2.imwrite(str(out / "fig2_procedures.png"), row)
    print("  Fig 2: 4 procedures comparison")


def make_figure_3_intensity(face: FaceLandmarks, image: np.ndarray, out: Path) -> None:
    """Figure 3: Intensity sweep for rhinoplasty."""
    s = 180
    intensities = [0, 20, 40, 60, 80, 100]
    panels = []

    _, _, orig_wf = generate_conditioning(face, 512, 512)

    for intensity in intensities:
        if intensity == 0:
            panels.append(put_label(resize_sq(image, s), "I=0 (original)"))
            continue

        manip = apply_procedure_preset(face, "rhinoplasty", float(intensity), 512)
        mask = generate_surgical_mask(face, "rhinoplasty", 512, 512)
        tps = warp_image_tps(image, face.pixel_coords, manip.pixel_coords)
        blended = mask_composite(tps, image, mask)
        panels.append(put_label(resize_sq(blended, s), f"I={intensity}"))

    row = np.hstack(panels)

    title_h = 30
    title_bar = np.zeros((title_h, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        title_bar,
        "Rhinoplasty Intensity Sweep (0-100)",
        (10, 22),
        FONT,
        0.45,
        ACCENT,
        1,
        cv2.LINE_AA,
    )
    figure = np.vstack([title_bar, row])
    cv2.imwrite(str(out / "fig3_intensity.png"), figure)
    print("  Fig 3: Intensity sweep")


def make_figure_4_multi_face(
    images: list[tuple[str, np.ndarray, FaceLandmarks]],
    out: Path,
    pipe: LandmarkDiffPipeline | None = None,
) -> None:
    """Figure 4: Multiple faces x all procedures grid."""
    s = 160
    procedures = [
        "rhinoplasty",
        "blepharoplasty",
        "rhytidectomy",
        "orthognathic",
        "brow_lift",
        "mentoplasty",
    ]
    rows = []

    for name, image, face in images:
        row_panels = [put_label(resize_sq(image, s), name, "bottom")]

        for proc in procedures:
            manip = apply_procedure_preset(face, proc, 65.0, 512)
            mask = generate_surgical_mask(face, proc, 512, 512)
            tps = warp_image_tps(image, face.pixel_coords, manip.pixel_coords)
            blended = mask_composite(tps, image, mask)

            if pipe and pipe.is_loaded and pipe.mode != "tps":
                try:
                    result = pipe.generate(image, proc, 65.0, seed=42)
                    panel = resize_sq(result["output"], s)
                except Exception:
                    panel = resize_sq(blended, s)
            else:
                panel = resize_sq(blended, s)

            panel = put_label(panel, proc[:8], "bottom")
            row_panels.append(panel)

        rows.append(np.hstack(row_panels))

    # Header
    header_h = 30
    w = rows[0].shape[1]
    header = np.zeros((header_h, w, 3), dtype=np.uint8)
    labels = ["Original", "Rhinoplasty", "Blepharoplasty", "Rhytidectomy", "Orthognathic"]
    col_w = s
    for i, label in enumerate(labels):
        x = i * col_w + 10
        cv2.putText(header, label, (x, 22), FONT, 0.4, ACCENT, 1, cv2.LINE_AA)

    grid = np.vstack([header] + rows)
    cv2.imwrite(str(out / "fig4_multi_face.png"), grid)
    print(f"  Fig 4: Multi-face grid ({len(images)} faces)")


def make_figure_5_training_data(out: Path) -> None:
    """Figure 5: Training data samples."""
    s = 140
    pair_dir = Path("data/synthetic_pairs")
    if not pair_dir.exists():
        print("  Fig 5: Skipped (no synthetic_pairs)")
        return

    rows = []
    for i in range(min(5, len(list(pair_dir.glob("*_input.png"))))):
        prefix = f"{i:06d}"
        panels = []
        for label, suffix in [
            ("Input", "input"),
            ("Target", "target"),
            ("Conditioning", "conditioning"),
            ("Canny", "canny"),
            ("Mask", "mask"),
        ]:
            fpath = pair_dir / f"{prefix}_{suffix}.png"
            if fpath.exists():
                img = cv2.imread(str(fpath))
                if img is not None:
                    panels.append(put_label(resize_sq(img, s), label, "bottom"))

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

        title_h = 30
        w = padded[0].shape[1]
        title = np.zeros((title_h, w, 3), dtype=np.uint8)
        cv2.putText(
            title,
            "Synthetic Training Pairs (TPS warp + clinical augmentation)",
            (10, 22),
            FONT,
            0.45,
            ACCENT,
            1,
            cv2.LINE_AA,
        )
        grid = np.vstack([title] + padded)
        cv2.imwrite(str(out / "fig5_training_data.png"), grid)
        print("  Fig 5: Training data samples")


def make_figure_6_before_after(
    images: list[tuple[str, np.ndarray, FaceLandmarks]],
    out: Path,
    pipe: LandmarkDiffPipeline | None = None,
) -> None:
    """Figure 6: Clean before/after pairs for each procedure."""
    s = 256
    procedures = ["rhinoplasty", "blepharoplasty"]

    for proc in procedures:
        rows = []
        for _name, image, face in images[:3]:
            manip = apply_procedure_preset(face, proc, 60.0, 512)
            mask = generate_surgical_mask(face, proc, 512, 512)
            tps = warp_image_tps(image, face.pixel_coords, manip.pixel_coords)
            blended = mask_composite(tps, image, mask)

            if pipe and pipe.is_loaded and pipe.mode != "tps":
                try:
                    result = pipe.generate(image, proc, 60.0, seed=42)
                    after = result["output"]
                except Exception:
                    after = blended
            else:
                after = blended

            before = resize_sq(image, s)
            after_r = resize_sq(after, s)

            # Arrow separator
            arrow = np.zeros((s, 40, 3), dtype=np.uint8)
            cv2.arrowedLine(arrow, (5, s // 2), (35, s // 2), WHITE, 2, tipLength=0.4)

            pair = np.hstack(
                [
                    put_label(before, "Before"),
                    arrow,
                    put_label(after_r, "After"),
                ]
            )
            rows.append(pair)

        if rows:
            grid = np.vstack(rows)
            cv2.imwrite(str(out / f"fig6_before_after_{proc}.png"), grid)
            print(f"  Fig 6: Before/after ({proc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="PI Demo - LandmarkDiff results")
    parser.add_argument("--mode", default="tps", choices=["tps", "img2img", "controlnet"])
    parser.add_argument("--output", default="scripts/final_output/pi_demo")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load faces
    ffhq = sorted(Path("data/ffhq_samples").glob("*.png"))[:6]
    if not ffhq:
        print("ERROR: No FFHQ samples in data/ffhq_samples/")
        sys.exit(1)

    print(f"Loading {len(ffhq)} faces...")
    faces = []
    for f in ffhq:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.resize(img, (512, 512))
        face = extract_landmarks(img)
        if face is not None:
            faces.append((f.stem, img, face))
    print(f"  {len(faces)} faces detected\n")

    # Load diffusion pipeline if requested
    pipe = None
    if args.mode != "tps":
        pipe = LandmarkDiffPipeline(mode=args.mode)
        pipe.load()
        print()

    print("Generating figures...")
    t0 = time.time()

    name, image, face = faces[0]

    make_figure_1_pipeline(face, image, out)
    make_figure_2_procedures(face, image, out, pipe)
    make_figure_3_intensity(face, image, out)
    make_figure_4_multi_face(faces, out, pipe)
    make_figure_5_training_data(out)
    make_figure_6_before_after(faces, out, pipe)

    elapsed = time.time() - t0
    print(f"\nAll figures saved to {out}/ ({elapsed:.1f}s)")
    print(f"Mode: {args.mode}")


if __name__ == "__main__":
    main()
