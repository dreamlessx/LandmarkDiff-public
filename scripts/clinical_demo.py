"""Interactive clinical demonstration script.

Processes a patient photo through the full LandmarkDiff pipeline,
generating a professional report with multiple procedure visualizations,
intensity variants, and quality metrics.

Usage:
    # Quick demo: single procedure
    python scripts/clinical_demo.py patient_photo.jpg

    # Full report: all procedures at multiple intensities
    python scripts/clinical_demo.py patient_photo.jpg --full-report

    # Specific procedure with displacement model
    python scripts/clinical_demo.py patient_photo.jpg \
        --procedure rhinoplasty \
        --displacement-model data/displacement_model.npz

    # Custom output directory
    python scripts/clinical_demo.py patient_photo.jpg \
        --output demo_output --full-report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarkdiff.evaluation import (
    classify_fitzpatrick_ita,
    compute_lpips,
    compute_ssim,
)
from landmarkdiff.inference import estimate_face_view, mask_composite
from landmarkdiff.landmarks import (
    FaceLandmarks,
    extract_landmarks,
    render_landmark_image,
    visualize_landmarks,
)
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps

PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]

PROCEDURE_DESCRIPTIONS = {
    "rhinoplasty": "Nose reshaping (rhinoplasty)",
    "blepharoplasty": "Eyelid surgery (blepharoplasty)",
    "rhytidectomy": "Facelift (rhytidectomy)",
    "orthognathic": "Jaw surgery (orthognathic)",
    "brow_lift": "Brow lift (forehead lift)",
    "mentoplasty": "Chin surgery (mentoplasty)",
}


def create_header_panel(
    text: str,
    width: int,
    height: int = 60,
    bg_color: tuple = (40, 40, 40),
    text_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """Create a text header panel."""
    panel = np.full((height, width, 3), bg_color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.9, 2)[0]
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2
    cv2.putText(panel, text, (x, y), font, 0.9, text_color, 2, cv2.LINE_AA)
    return panel


def create_label_panel(
    text: str,
    width: int,
    height: int = 30,
    bg_color: tuple = (30, 30, 30),
    text_color: tuple = (200, 200, 200),
) -> np.ndarray:
    """Create a small label bar."""
    panel = np.full((height, width, 3), bg_color, dtype=np.uint8)
    cv2.putText(panel, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    return panel


def generate_procedure_result(
    image: np.ndarray,
    face: FaceLandmarks,
    procedure: str,
    intensity: float,
    displacement_model_path: str | None = None,
) -> dict:
    """Generate a single procedure outcome."""
    t0 = time.time()

    manip = apply_procedure_preset(
        face,
        procedure,
        intensity,
        image_size=512,
        displacement_model_path=displacement_model_path,
    )

    mask = generate_surgical_mask(face, procedure, 512, 512)
    conditioning = render_landmark_image(manip, 512, 512)
    warped = warp_image_tps(image, face.pixel_coords, manip.pixel_coords)
    composited = mask_composite(warped, image, mask)

    elapsed = time.time() - t0

    # Compute metrics
    ssim_val = compute_ssim(composited, image)
    lpips_val = compute_lpips(composited, image)

    return {
        "output": composited,
        "warped": warped,
        "mask": mask,
        "conditioning": conditioning,
        "ssim": float(ssim_val),
        "lpips": float(lpips_val),
        "elapsed": elapsed,
    }


def create_single_procedure_report(
    image: np.ndarray,
    face: FaceLandmarks,
    procedure: str,
    intensities: list[float],
    output_dir: Path,
    displacement_model_path: str | None = None,
) -> dict:
    """Generate a single-procedure report with multiple intensities."""
    size = 512
    results = {}

    for intensity in intensities:
        result = generate_procedure_result(
            image,
            face,
            procedure,
            intensity,
            displacement_model_path=displacement_model_path,
        )
        results[intensity] = result

    # Create comparison figure
    # Row 1: Original + intensity variants
    panels_row1 = [image.copy()]
    labels_row1 = ["Original"]
    for intensity in intensities:
        panels_row1.append(results[intensity]["output"])
        labels_row1.append(f"{intensity:.0f}%")

    # Add labels
    labeled_panels = []
    for panel, label in zip(panels_row1, labels_row1, strict=False):
        lbl = create_label_panel(label, size)
        labeled_panels.append(np.vstack([lbl, panel]))

    comparison = np.hstack(labeled_panels)

    # Add header
    header = create_header_panel(
        PROCEDURE_DESCRIPTIONS.get(procedure, procedure),
        comparison.shape[1],
    )
    final = np.vstack([header, comparison])

    # Row 2: Intermediates for the middle intensity
    mid_idx = len(intensities) // 2
    mid_result = results[intensities[mid_idx]]
    mask_vis = cv2.cvtColor((mid_result["mask"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    intermediates = np.hstack(
        [
            np.vstack(
                [create_label_panel("Landmarks", size), visualize_landmarks(image, face, radius=2)]
            ),
            np.vstack([create_label_panel("Conditioning", size), mid_result["conditioning"]]),
            np.vstack([create_label_panel("Surgical Mask", size), mask_vis]),
            np.vstack([create_label_panel("TPS Warp", size), mid_result["warped"]]),
        ]
    )
    final = np.vstack([final, intermediates])

    cv2.imwrite(str(output_dir / f"{procedure}_report.png"), final)

    # Save individual outputs
    for intensity in intensities:
        cv2.imwrite(
            str(output_dir / f"{procedure}_{intensity:.0f}pct.png"),
            results[intensity]["output"],
        )

    return {
        procedure: {
            str(int(k)): {
                "ssim": v["ssim"],
                "lpips": v["lpips"],
                "elapsed": round(v["elapsed"], 3),
            }
            for k, v in results.items()
        }
    }


def create_full_report(
    image: np.ndarray,
    face: FaceLandmarks,
    procedures: list[str],
    output_dir: Path,
    displacement_model_path: str | None = None,
) -> dict:
    """Generate a full report covering all procedures."""
    intensities = [25, 50, 75, 100]
    all_metrics = {}

    # Generate per-procedure reports
    for proc in procedures:
        print(f"  Generating {proc}...")
        metrics = create_single_procedure_report(
            image,
            face,
            proc,
            intensities,
            output_dir,
            displacement_model_path=displacement_model_path,
        )
        all_metrics.update(metrics)

    # Create overview grid: procedures (rows) x intensities (columns)
    size = 256  # smaller for overview
    rows = []
    for proc in procedures:
        row_panels = []
        # Procedure label
        proc_label = np.full((size, 120, 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(
            proc_label,
            proc[:12],
            (5, size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        row_panels.append(proc_label)

        for intensity in intensities:
            result = generate_procedure_result(
                image,
                face,
                proc,
                intensity,
                displacement_model_path=displacement_model_path,
            )
            panel = cv2.resize(result["output"], (size, size))
            row_panels.append(panel)

        rows.append(np.hstack(row_panels))

    # Header row
    header_panels = [np.full((30, 120, 3), (40, 40, 40), dtype=np.uint8)]
    for intensity in intensities:
        h = create_label_panel(f"{intensity:.0f}%", size, 30)
        header_panels.append(h)
    header_row = np.hstack(header_panels)

    # Title
    total_width = header_row.shape[1]
    title = create_header_panel("Surgical Outcome Predictions — Overview", total_width, 50)

    overview = np.vstack([title, header_row] + rows)

    # Add original image on the left
    orig_resized = cv2.resize(image, (size, size))
    orig_labeled = np.vstack(
        [
            create_label_panel("Original", size, 30),
            orig_resized,
        ]
    )
    # Pad to match overview height
    pad_h = overview.shape[0] - orig_labeled.shape[0]
    if pad_h > 0:
        orig_labeled = np.vstack(
            [
                orig_labeled,
                np.full((pad_h, size, 3), (20, 20, 20), dtype=np.uint8),
            ]
        )

    full_overview = np.hstack([orig_labeled, overview])
    cv2.imwrite(str(output_dir / "overview_grid.png"), full_overview)

    return all_metrics


def run_demo(
    image_path: str,
    procedure: str = "rhinoplasty",
    intensity: float = 65.0,
    output_dir: str = "demo_output",
    full_report: bool = False,
    displacement_model_path: str | None = None,
    seed: int = 42,
) -> None:
    """Run the clinical demonstration."""
    img_path = Path(image_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not img_path.exists():
        print(f"ERROR: Image not found: {img_path}")
        sys.exit(1)

    # Load and prepare image
    print(f"Loading image: {img_path}")
    img = cv2.imread(str(img_path))
    if img is None:
        print("ERROR: Cannot read image")
        sys.exit(1)

    img_512 = cv2.resize(img, (512, 512))

    # Extract landmarks
    print("Extracting facial landmarks...")
    face = extract_landmarks(img_512)
    if face is None:
        print("ERROR: No face detected in image")
        sys.exit(1)

    n_landmarks = len(face.landmarks)
    print(f"  Detected {n_landmarks} landmarks (confidence: {face.confidence:.2f})")

    # View analysis
    view = estimate_face_view(face)
    print(f"  View: {view.get('view', '?')} (yaw: {view.get('yaw_deg', 0):.1f}deg)")
    if abs(view.get("yaw_deg", 0)) > 30:
        print("  WARNING: Non-frontal face — results may be less accurate")

    # Fitzpatrick classification
    try:
        fitz = classify_fitzpatrick_ita(img_512)
        print(f"  Estimated Fitzpatrick type: {fitz}")
    except Exception:
        fitz = "?"

    # Save landmark visualization
    lm_vis = visualize_landmarks(img_512, face, radius=2)
    cv2.imwrite(str(out / "landmarks.png"), lm_vis)

    # Generate report
    report_data = {
        "image": str(img_path),
        "landmarks": n_landmarks,
        "confidence": float(face.confidence),
        "fitzpatrick": fitz,
        "view": view,
    }

    if full_report:
        print("\nGenerating full report (all procedures)...")
        metrics = create_full_report(
            img_512,
            face,
            PROCEDURES,
            out,
            displacement_model_path=displacement_model_path,
        )
        report_data["metrics"] = metrics
    else:
        print(f"\nGenerating {procedure} at {intensity}% intensity...")
        intensities = [25, 50, 75, 100] if intensity == 65.0 else [intensity]
        metrics = create_single_procedure_report(
            img_512,
            face,
            procedure,
            intensities,
            out,
            displacement_model_path=displacement_model_path,
        )
        report_data["metrics"] = metrics

    # Save report JSON
    report_path = out / "report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    # Print summary
    print(f"\n{'=' * 60}")
    print("  Clinical Demo Report")
    print(f"{'=' * 60}")
    print(f"  Input: {img_path.name}")
    print(f"  Fitzpatrick: Type {fitz}")
    print(f"  Face view: {view.get('view', '?')}")
    print()
    for proc, proc_metrics in report_data["metrics"].items():
        print(f"  {PROCEDURE_DESCRIPTIONS.get(proc, proc)}:")
        for intensity_key, m in proc_metrics.items():
            print(f"    {intensity_key}%: SSIM={m['ssim']:.3f}, LPIPS={m['lpips']:.3f}")
    print()
    print(f"  Output directory: {out}")
    print(f"  Report: {report_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical demonstration")
    parser.add_argument("image", help="Path to patient face image")
    parser.add_argument("--procedure", default="rhinoplasty", choices=PROCEDURES)
    parser.add_argument(
        "--intensity", type=float, default=65.0, help="Intensity percentage (0-100)"
    )
    parser.add_argument("--output", default="demo_output", help="Output directory")
    parser.add_argument(
        "--full-report", action="store_true", help="Generate report for all procedures"
    )
    parser.add_argument(
        "--displacement-model", default=None, help="Path to data-driven displacement model (.npz)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_demo(
        args.image,
        args.procedure,
        args.intensity,
        args.output,
        args.full_report,
        args.displacement_model,
        args.seed,
    )
