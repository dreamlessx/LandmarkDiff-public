"""Demo - full pipeline visualization on a single face image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.conditioning import generate_conditioning
from landmarkdiff.landmarks import (
    extract_landmarks,
    load_image,
    render_landmark_image,
    visualize_landmarks,
)
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask, mask_to_3channel


def run_demo(
    image_path: str,
    procedure: str = "rhinoplasty",
    intensity: float = 50.0,
    output_dir: str = "scripts/demo_output",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load image
    print(f"Loading {image_path}...")
    image = load_image(image_path)
    h, w = image.shape[:2]
    print(f"  Image size: {w}x{h}")

    # 2. Extract landmarks
    print("Extracting 478 facial landmarks...")
    face = extract_landmarks(image)
    if face is None:
        print("ERROR: No face detected in image.")
        sys.exit(1)
    print(f"  Detected face with {len(face.landmarks)} landmarks")

    # 3. Visualize original landmarks on image
    annotated = visualize_landmarks(image, face, radius=2)
    cv2.imwrite(str(out / "01_landmarks_on_face.png"), annotated)
    print("  Saved: 01_landmarks_on_face.png")

    # 4. Render landmark dots on black canvas
    landmark_img = render_landmark_image(face)
    cv2.imwrite(str(out / "02_landmark_dots.png"), landmark_img)
    print("  Saved: 02_landmark_dots.png")

    # 5. Generate conditioning signals (wireframe + canny)
    print("Generating conditioning signals...")
    _, canny, wireframe = generate_conditioning(face)
    cv2.imwrite(str(out / "03_wireframe.png"), wireframe)
    cv2.imwrite(str(out / "04_canny_edges.png"), canny)
    print("  Saved: 03_wireframe.png, 04_canny_edges.png")

    # 6. Apply surgical manipulation
    print(f"Applying {procedure} at intensity {intensity}...")
    manipulated = apply_procedure_preset(face, procedure, intensity)

    # 7. Render manipulated conditioning
    manip_landmark = render_landmark_image(manipulated)
    _, manip_canny, manip_wireframe = generate_conditioning(manipulated)
    cv2.imwrite(str(out / "05_manipulated_landmarks.png"), manip_landmark)
    cv2.imwrite(str(out / "06_manipulated_wireframe.png"), manip_wireframe)
    cv2.imwrite(str(out / "07_manipulated_canny.png"), manip_canny)
    print("  Saved: 05-07 manipulated conditioning")

    # 8. Generate surgical mask
    print("Generating surgical mask...")
    mask = generate_surgical_mask(face, procedure)
    mask_vis = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(out / "08_surgical_mask.png"), mask_vis)
    print("  Saved: 08_surgical_mask.png")

    # 9. Overlay mask on original image
    mask_overlay = image.copy()
    mask_3ch = mask_to_3channel(mask)
    red_tint = np.zeros_like(image, dtype=np.float32)
    red_tint[:, :, 2] = 255.0  # red channel
    mask_overlay = (
        image.astype(np.float32) * (1 - mask_3ch * 0.4) + red_tint * (mask_3ch * 0.4)
    ).astype(np.uint8)
    cv2.imwrite(str(out / "09_mask_overlay.png"), mask_overlay)
    print("  Saved: 09_mask_overlay.png")

    # 10. Side-by-side comparison: original vs manipulated conditioning
    orig_cond = np.hstack([wireframe, canny])
    manip_cond = np.hstack([manip_wireframe, manip_canny])
    comparison = np.vstack([orig_cond, manip_cond])
    cv2.imwrite(str(out / "10_before_after_conditioning.png"), comparison)
    print("  Saved: 10_before_after_conditioning.png")

    # 11. Full pipeline summary composite
    # Resize all to same height for composite
    target_h = 256

    def resize_to_h(img: np.ndarray, target: int) -> np.ndarray:
        scale = target / img.shape[0]
        new_w = int(img.shape[1] * scale)
        resized = cv2.resize(img, (new_w, target))
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        return resized

    panels = [
        resize_to_h(image, target_h),
        resize_to_h(annotated, target_h),
        resize_to_h(cv2.cvtColor(wireframe, cv2.COLOR_GRAY2BGR), target_h),
        resize_to_h(cv2.cvtColor(manip_wireframe, cv2.COLOR_GRAY2BGR), target_h),
        resize_to_h(mask_overlay, target_h),
    ]
    composite = np.hstack(panels)
    cv2.imwrite(str(out / "11_pipeline_summary.png"), composite)
    print("  Saved: 11_pipeline_summary.png")

    print(f"\nDone. All outputs in {out}/")
    print(f"Pipeline: Input -> Landmarks -> Wireframe -> Manipulate ({procedure}) -> Mask")
    print("Next step: ControlNet inference (requires GPU + model weights)")


def run_synthetic_demo(output_dir: str = "scripts/demo_output") -> None:
    """Run demo with a synthetic face when no image is provided."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("No image provided - running synthetic landmark demo...")
    print()

    # Create synthetic face landmarks (rough oval face shape)
    rng = np.random.default_rng(42)
    landmarks = np.zeros((478, 3), dtype=np.float32)

    # Place landmarks in a rough face layout
    for i in range(478):
        angle = (i / 478) * 2 * np.pi
        r = 0.15 + 0.05 * np.sin(3 * angle) + rng.uniform(-0.02, 0.02)
        landmarks[i, 0] = 0.5 + r * np.cos(angle)
        landmarks[i, 1] = 0.5 + r * 1.2 * np.sin(angle)
        landmarks[i, 2] = rng.uniform(-0.01, 0.01)

    # Override specific regions for more realistic layout
    # Nose center
    for idx in [1, 2, 4, 5, 6, 19, 94, 168, 195, 197]:
        if idx < 478:
            landmarks[idx] = [0.5 + rng.uniform(-0.01, 0.01), 0.48 + rng.uniform(-0.02, 0.02), 0.0]

    # Eyes
    for idx in [33, 133, 159, 145]:
        if idx < 478:
            landmarks[idx] = [0.38 + rng.uniform(-0.01, 0.01), 0.40 + rng.uniform(-0.01, 0.01), 0.0]
    for idx in [362, 263, 386, 374]:
        if idx < 478:
            landmarks[idx] = [0.62 + rng.uniform(-0.01, 0.01), 0.40 + rng.uniform(-0.01, 0.01), 0.0]

    # Mouth
    for idx in [61, 291, 78, 308, 13, 14, 17, 84]:
        if idx < 478:
            landmarks[idx] = [0.5 + rng.uniform(-0.03, 0.03), 0.60 + rng.uniform(-0.01, 0.01), 0.0]

    from landmarkdiff.conditioning import generate_conditioning
    from landmarkdiff.landmarks import FaceLandmarks
    from landmarkdiff.manipulation import apply_procedure_preset
    from landmarkdiff.masking import generate_surgical_mask

    face = FaceLandmarks(landmarks=landmarks, image_width=512, image_height=512, confidence=0.9)

    # Render original
    landmark_img = render_landmark_image(face, 512, 512)
    cv2.imwrite(str(out / "01_synthetic_landmarks.png"), landmark_img)

    _, canny, wireframe = generate_conditioning(face, 512, 512)
    cv2.imwrite(str(out / "02_synthetic_wireframe.png"), wireframe)
    cv2.imwrite(str(out / "03_synthetic_canny.png"), canny)

    # Apply each procedure
    for proc in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
        manip = apply_procedure_preset(face, proc, intensity=70.0, image_size=512)
        manip_img = render_landmark_image(manip, 512, 512)
        _, manip_canny, manip_wf = generate_conditioning(manip, 512, 512)
        mask = generate_surgical_mask(face, proc, 512, 512)

        cv2.imwrite(str(out / f"04_{proc}_landmarks.png"), manip_img)
        cv2.imwrite(str(out / f"05_{proc}_wireframe.png"), manip_wf)
        cv2.imwrite(str(out / f"06_{proc}_mask.png"), (mask * 255).astype(np.uint8))

        # Side-by-side: original wireframe vs manipulated
        comparison = np.hstack([wireframe, manip_wf])
        cv2.imwrite(str(out / f"07_{proc}_comparison.png"), comparison)
        print(f"  {proc}: landmarks, wireframe, mask, comparison saved")

    print(f"\nAll outputs in {out}/")
    print("Run with a real face image for full pipeline: python scripts/demo.py <image.jpg>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LandmarkDiff pipeline demo")
    parser.add_argument("image", nargs="?", help="Path to face image")
    parser.add_argument(
        "--procedure",
        default="rhinoplasty",
        choices=["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"],
    )
    parser.add_argument(
        "--intensity", type=float, default=50.0, help="Manipulation intensity 0-100"
    )
    parser.add_argument("--output", default="scripts/demo_output", help="Output directory")
    args = parser.parse_args()

    if args.image:
        run_demo(args.image, args.procedure, args.intensity, args.output)
    else:
        run_synthetic_demo(args.output)
