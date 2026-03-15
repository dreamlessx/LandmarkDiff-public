"""Split comparison images into before/after pairs for ControlNet training."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from landmarkdiff.landmarks import FaceLandmarks, extract_landmarks, render_landmark_image


def split_comparison_image(img: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Try horizontal and vertical splits, return all valid (before, after) candidates."""
    h, w = img.shape[:2]
    candidates = []

    if w < 100 or h < 100:
        return candidates

    aspect = w / h

    # Strategy 1: Horizontal split (side by side) - most common
    if aspect > 1.2:
        mid = w // 2
        gap = max(2, w // 100)
        left = img[:, : mid - gap]
        right = img[:, mid + gap :]
        if left.shape[1] > 80 and right.shape[1] > 80:
            target_h = min(left.shape[0], right.shape[0])
            target_w = min(left.shape[1], right.shape[1])
            candidates.append(
                (
                    cv2.resize(left, (target_w, target_h)),
                    cv2.resize(right, (target_w, target_h)),
                )
            )

    # Strategy 2: Vertical split (top/bottom)
    if aspect < 0.85:
        mid = h // 2
        gap = max(2, h // 100)
        top = img[: mid - gap, :]
        bottom = img[mid + gap :, :]
        if top.shape[0] > 80 and bottom.shape[0] > 80:
            target_h = min(top.shape[0], bottom.shape[0])
            target_w = min(top.shape[1], bottom.shape[1])
            candidates.append(
                (
                    cv2.resize(top, (target_w, target_h)),
                    cv2.resize(bottom, (target_w, target_h)),
                )
            )

    # Strategy 3: For ~square images, try horizontal anyway
    if 0.85 <= aspect <= 1.2:
        mid = w // 2
        gap = max(2, w // 100)
        left = img[:, : mid - gap]
        right = img[:, mid + gap :]
        if left.shape[1] > 80 and right.shape[1] > 80:
            target_h = min(left.shape[0], right.shape[0])
            target_w = min(left.shape[1], right.shape[1])
            candidates.append(
                (
                    cv2.resize(left, (target_w, target_h)),
                    cv2.resize(right, (target_w, target_h)),
                )
            )

    return candidates


def validate_and_extract(
    before: np.ndarray,
    after: np.ndarray,
    target_size: int = 512,
) -> tuple[FaceLandmarks | None, FaceLandmarks | None]:
    """Validate face detection in both images and extract landmarks."""
    before_resized = cv2.resize(before, (target_size, target_size))
    after_resized = cv2.resize(after, (target_size, target_size))

    face_before = extract_landmarks(before_resized)
    face_after = extract_landmarks(after_resized)

    return face_before, face_after


def compute_displacement(
    before: FaceLandmarks,
    after: FaceLandmarks,
) -> np.ndarray:
    """Per-landmark (dx, dy) in normalized coords, shape (478, 2)."""
    before_pts = before.landmarks[:, :2]
    after_pts = after.landmarks[:, :2]
    return after_pts - before_pts


def main():
    parser = argparse.ArgumentParser(description="Process real surgery before/after images")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/real_surgery_pairs/raw",
        help="Directory with raw comparison images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/real_surgery_pairs",
        help="Output directory for processed pairs",
    )
    parser.add_argument("--size", type=int, default=512, help="Target image size")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output)

    pairs_dir = output_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    all_displacements = {}
    all_pairs = []
    pair_idx = 0

    procedures = [
        "rhinoplasty",
        "blepharoplasty",
        "rhytidectomy",
        "orthognathic",
        "brow_lift",
        "mentoplasty",
    ]

    for proc in procedures:
        proc_dir = raw_dir / proc
        if not proc_dir.exists():
            print(f"Skipping {proc} (no directory)")
            continue

        images = sorted(proc_dir.glob("*"))
        images = [
            f for f in images if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
        print(f"\n=== {proc}: {len(images)} raw images ===")

        proc_displacements = []
        proc_valid = 0

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            candidates = split_comparison_image(img)
            if not candidates:
                continue

            for before, after in candidates:
                face_before, face_after = validate_and_extract(before, after, args.size)

                if face_before is None or face_after is None:
                    continue

                # Compute displacement
                disp = compute_displacement(face_before, face_after)
                proc_displacements.append(disp)

                # Render meshes
                before_img = cv2.resize(before, (args.size, args.size))
                after_img = cv2.resize(after, (args.size, args.size))
                before_mesh = render_landmark_image(face_before, args.size, args.size)
                after_mesh = render_landmark_image(face_after, args.size, args.size)

                # Save as ControlNet training pair:
                # input = after_mesh (post-surgery mesh = conditioning)
                # target = after_img (post-surgery face = generation target)
                input_path = pairs_dir / f"{pair_idx:05d}_{proc}_input.png"
                target_path = pairs_dir / f"{pair_idx:05d}_{proc}_target.png"
                before_path = pairs_dir / f"{pair_idx:05d}_{proc}_before.png"
                before_mesh_path = pairs_dir / f"{pair_idx:05d}_{proc}_before_mesh.png"

                cv2.imwrite(str(input_path), after_mesh)
                cv2.imwrite(str(target_path), after_img)
                cv2.imwrite(str(before_path), before_img)
                cv2.imwrite(str(before_mesh_path), before_mesh)

                all_pairs.append(
                    {
                        "pair_id": pair_idx,
                        "procedure": proc,
                        "source_image": str(img_path),
                        "input_path": str(input_path),
                        "target_path": str(target_path),
                        "before_path": str(before_path),
                        "before_mesh_path": str(before_mesh_path),
                        "mean_displacement": float(np.mean(np.abs(disp))),
                        "max_displacement": float(np.max(np.abs(disp))),
                    }
                )

                pair_idx += 1
                proc_valid += 1

            if proc_valid % 20 == 0 and proc_valid > 0:
                print(f"  Processed {proc_valid} valid pairs...")

        print(f"  {proc}: {proc_valid} valid pairs from {len(images)} images")

        if proc_displacements:
            all_displacements[proc] = {
                "count": len(proc_displacements),
                "mean": np.mean(proc_displacements, axis=0).tolist(),
                "std": np.std(proc_displacements, axis=0).tolist(),
                "median": np.median(proc_displacements, axis=0).tolist(),
            }

    # Save metadata
    with open(output_dir / "pairs_metadata.json", "w") as f:
        json.dump(all_pairs, f, indent=2)

    # Save displacement statistics
    displacement_stats = {}
    for proc, data in all_displacements.items():
        displacement_stats[proc] = {
            "count": data["count"],
            "mean_abs_displacement": float(np.mean(np.abs(data["mean"]))),
            "max_abs_displacement": float(np.max(np.abs(data["mean"]))),
        }

    with open(output_dir / "displacement_stats.json", "w") as f:
        json.dump(displacement_stats, f, indent=2)

    print("\n=== Summary ===")
    print(f"Total valid pairs: {len(all_pairs)}")
    for proc in procedures:
        n = len([p for p in all_pairs if p["procedure"] == proc])
        if n > 0:
            stats = displacement_stats.get(proc, {})
            print(f"  {proc}: {n} pairs, mean_disp={stats.get('mean_abs_displacement', 0):.4f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
