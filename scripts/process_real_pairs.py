"""Split multi-panel clinical photos into before/after pairs for training."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.conditioning import generate_conditioning
from landmarkdiff.landmarks import extract_landmarks, render_landmark_image


def split_grid_image(image: np.ndarray, min_panel_size: int = 100) -> list[np.ndarray]:
    """Split grid image (2x1, 2x2, etc.) into panels using edge-based divider detection."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try common grid layouts
    panels = []

    # Detect vertical splits
    col_profile = np.mean(gray, axis=0)
    col_grad = np.abs(np.diff(col_profile))

    # Detect horizontal splits
    row_profile = np.mean(gray, axis=1)
    row_grad = np.abs(np.diff(row_profile))

    # Find strong vertical dividers
    v_threshold = np.mean(col_grad) + 2 * np.std(col_grad)
    v_splits = [0]
    i = 0
    while i < len(col_grad):
        if col_grad[i] > v_threshold:
            # Find the center of this divider region
            j = i
            while j < len(col_grad) and col_grad[j] > v_threshold * 0.5:
                j += 1
            mid = (i + j) // 2
            if mid - v_splits[-1] > min_panel_size:
                v_splits.append(mid)
            i = j
        else:
            i += 1
    v_splits.append(w)

    # Find strong horizontal dividers
    h_threshold = np.mean(row_grad) + 2 * np.std(row_grad)
    h_splits = [0]
    i = 0
    while i < len(row_grad):
        if row_grad[i] > h_threshold:
            j = i
            while j < len(row_grad) and row_grad[j] > h_threshold * 0.5:
                j += 1
            mid = (i + j) // 2
            if mid - h_splits[-1] > min_panel_size:
                h_splits.append(mid)
            i = j
        else:
            i += 1
    h_splits.append(h)

    # Extract panels
    for ri in range(len(h_splits) - 1):
        for ci in range(len(v_splits) - 1):
            y1, y2 = h_splits[ri], h_splits[ri + 1]
            x1, x2 = v_splits[ci], v_splits[ci + 1]

            panel = image[y1:y2, x1:x2]
            ph, pw = panel.shape[:2]

            if ph >= min_panel_size and pw >= min_panel_size:
                panels.append(panel)

    # If no splits found, return the whole image
    if len(panels) <= 1:
        # Try simple 2-column split
        mid_w = w // 2
        left = image[:, :mid_w]
        right = image[:, mid_w:]
        if left.shape[1] >= min_panel_size and right.shape[1] >= min_panel_size:
            return [left, right]
        return [image]

    return panels


def process_directory(
    input_dir: str = "data/real_pairs",
    output_dir: str = "data/real_processed",
) -> None:
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    procedures = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    total_processed = 0
    total_faces = 0

    for proc in procedures:
        proc_dir = inp / proc
        if not proc_dir.exists():
            continue

        proc_out = out / proc
        proc_out.mkdir(exist_ok=True)

        images = sorted(proc_dir.glob("*.*"))
        print(f"\n=== {proc}: {len(images)} images ===")

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # Split grid images into panels
            panels = split_grid_image(image)
            print(f"  {img_path.name}: {len(panels)} panels")

            for _pi, panel in enumerate(panels):
                # Resize to 512x512
                panel_512 = cv2.resize(panel, (512, 512))

                # Extract landmarks
                face = extract_landmarks(panel_512)
                if face is None:
                    continue

                total_faces += 1
                prefix = f"{total_processed:06d}"

                # Save panel
                cv2.imwrite(str(proc_out / f"{prefix}_face.png"), panel_512)

                # Save landmarks
                landmark_img = render_landmark_image(face, 512, 512)
                cv2.imwrite(str(proc_out / f"{prefix}_landmarks.png"), landmark_img)

                # Save conditioning
                _, canny, wireframe = generate_conditioning(face, 512, 512)
                cv2.imwrite(str(proc_out / f"{prefix}_wireframe.png"), wireframe)
                cv2.imwrite(str(proc_out / f"{prefix}_canny.png"), canny)

                # Save annotated
                from landmarkdiff.landmarks import visualize_landmarks

                annotated = visualize_landmarks(panel_512, face, radius=2)
                cv2.imwrite(str(proc_out / f"{prefix}_annotated.png"), annotated)

                total_processed += 1

    # Build gallery of processed real faces
    _build_gallery(out, total_processed)

    print(f"\nDone. {total_processed} panels with faces from {total_faces} detections")
    print(f"Output: {out}/")


def _build_gallery(out: Path, total: int) -> None:
    """Build a gallery grid of all processed real faces."""
    s = 128
    panels = []

    for i in range(min(total, 40)):
        prefix = f"{i:06d}"

        for proc_dir in out.iterdir():
            if not proc_dir.is_dir():
                continue
            face_path = proc_dir / f"{prefix}_face.png"
            if face_path.exists():
                img = cv2.imread(str(face_path))
                if img is not None:
                    panels.append(cv2.resize(img, (s, s)))
                break

    if not panels:
        return

    # Grid: 8 per row
    cols = 8
    rows_needed = (len(panels) + cols - 1) // cols
    while len(panels) < rows_needed * cols:
        panels.append(np.zeros((s, s, 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows_needed):
        row = np.hstack(panels[r * cols : (r + 1) * cols])
        grid_rows.append(row)

    grid = np.vstack(grid_rows)
    cv2.imwrite(str(out / "gallery_real_faces.png"), grid)
    print(f"  Gallery: {out / 'gallery_real_faces.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split multi-panel clinical photos into before/after pairs for training."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/real_pairs",
        help="Input directory containing procedure subdirectories (default: data/real_pairs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/real_processed",
        help="Output directory for processed images (default: data/real_processed)",
    )

    args = parser.parse_args()
    process_directory(input_dir=args.input, output_dir=args.output)
