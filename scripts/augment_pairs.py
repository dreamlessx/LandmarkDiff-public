"""Augment real before/after pairs (flip, jitter, rotate, crop, noise).

Same spatial transform to both images so landmark correspondence is preserved.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def augment_pair(
    before: np.ndarray,
    after: np.ndarray,
    augmentations_per_pair: int = 30,
    target_size: int = 512,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Generate augmented (before, after, aug_name) tuples from one pair."""
    results = []

    # 1. Horizontal flip
    results.append((
        cv2.flip(before, 1),
        cv2.flip(after, 1),
        "hflip",
    ))

    for i in range(augmentations_per_pair - 1):
        aug_name_parts = []
        b = before.copy()
        a = after.copy()

        # Random horizontal flip (50%)
        if random.random() < 0.5:
            b = cv2.flip(b, 1)
            a = cv2.flip(a, 1)
            aug_name_parts.append("hf")

        # Color jitter - same params for both
        if random.random() < 0.7:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.85, 1.15)
            saturation = random.uniform(0.8, 1.2)

            for img_ref in [b, a]:
                # Brightness
                hsv = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
                # Saturation
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
                img_ref[:] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Contrast
            for img_ref in [b, a]:
                mean = np.mean(img_ref)
                img_ref[:] = np.clip(
                    (img_ref.astype(np.float32) - mean) * contrast + mean, 0, 255
                ).astype(np.uint8)

            aug_name_parts.append(f"cj{brightness:.1f}")

        # Slight rotation (same angle for both)
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            h, w = b.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            b = cv2.warpAffine(b, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            a = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            aug_name_parts.append(f"rot{angle:.0f}")

        # Random crop + resize (same crop for both)
        if random.random() < 0.4:
            h, w = b.shape[:2]
            crop_frac = random.uniform(0.85, 0.95)
            ch, cw = int(h * crop_frac), int(w * crop_frac)
            y = random.randint(0, h - ch)
            x = random.randint(0, w - cw)
            b = cv2.resize(b[y:y+ch, x:x+cw], (target_size, target_size))
            a = cv2.resize(a[y:y+ch, x:x+cw], (target_size, target_size))
            aug_name_parts.append(f"crop{crop_frac:.2f}")

        # Gaussian noise (different for each image - represents sensor noise)
        if random.random() < 0.3:
            sigma = random.uniform(2, 8)
            noise_b = np.random.randn(*b.shape) * sigma
            noise_a = np.random.randn(*a.shape) * sigma
            b = np.clip(b.astype(np.float32) + noise_b, 0, 255).astype(np.uint8)
            a = np.clip(a.astype(np.float32) + noise_a, 0, 255).astype(np.uint8)
            aug_name_parts.append(f"noise{sigma:.0f}")

        # Gamma correction (same for both)
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.3)
            table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                              for i in np.arange(256)]).astype(np.uint8)
            b = cv2.LUT(b, table)
            a = cv2.LUT(a, table)
            aug_name_parts.append(f"gam{gamma:.1f}")

        # Ensure target size
        if b.shape[:2] != (target_size, target_size):
            b = cv2.resize(b, (target_size, target_size))
            a = cv2.resize(a, (target_size, target_size))

        aug_name = "_".join(aug_name_parts) if aug_name_parts else f"v{i}"
        results.append((b, a, aug_name))

    return results


def main():
    parser = argparse.ArgumentParser(description="Augment surgery before/after pairs")
    parser.add_argument("--pairs_dir", type=str, default="data/real_surgery_pairs/pairs",
                        help="Directory with validated pairs")
    parser.add_argument("--metadata", type=str, default="data/real_surgery_pairs/pairs_metadata.json",
                        help="Pairs metadata JSON")
    parser.add_argument("--output", type=str, default="data/real_surgery_pairs/augmented",
                        help="Output directory")
    parser.add_argument("--target_per_proc", type=int, default=50000,
                        help="Target pairs per procedure")
    parser.add_argument("--size", type=int, default=512, help="Image size")
    args = parser.parse_args()

    with open(args.metadata) as f:
        metadata = json.load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by procedure
    by_proc = {}
    for pair in metadata:
        proc = pair["procedure"]
        if proc not in by_proc:
            by_proc[proc] = []
        by_proc[proc].append(pair)

    all_augmented = []

    for proc, pairs in by_proc.items():
        print(f"\n=== {proc}: {len(pairs)} original pairs ===")
        if len(pairs) == 0:
            continue

        # Calculate augmentations needed per pair
        augs_per_pair = max(1, args.target_per_proc // len(pairs))
        augs_per_pair = min(augs_per_pair, 100)  # cap at 100x
        print(f"  Augmenting {augs_per_pair}x per pair -> target ~{len(pairs) * augs_per_pair}")

        proc_dir = output_dir / proc
        proc_dir.mkdir(parents=True, exist_ok=True)

        aug_idx = 0
        for pair_i, pair in enumerate(pairs):
            # Load the before and after images
            before_path = pair.get("before_path")
            target_path = pair.get("target_path")

            if not before_path or not target_path:
                continue

            before = cv2.imread(before_path)
            after = cv2.imread(target_path)

            if before is None or after is None:
                continue

            # Generate augmentations
            augmented = augment_pair(before, after, augs_per_pair, args.size)

            for b_aug, a_aug, aug_name in augmented:
                before_out = proc_dir / f"{aug_idx:06d}_before.png"
                after_out = proc_dir / f"{aug_idx:06d}_after.png"

                cv2.imwrite(str(before_out), b_aug)
                cv2.imwrite(str(after_out), a_aug)

                all_augmented.append({
                    "idx": aug_idx,
                    "procedure": proc,
                    "source_pair": pair.get("pair_id", pair_i),
                    "augmentation": aug_name,
                    "before_path": str(before_out),
                    "after_path": str(after_out),
                })
                aug_idx += 1

            if (pair_i + 1) % 10 == 0:
                print(f"  {pair_i + 1}/{len(pairs)} pairs processed, {aug_idx} total augmented")

        print(f"  {proc}: {aug_idx} augmented pairs")

    # Save metadata
    with open(output_dir / "augmented_metadata.json", "w") as f:
        json.dump(all_augmented, f, indent=2)

    print(f"\nTotal augmented pairs: {len(all_augmented)}")
    for proc in by_proc:
        n = len([a for a in all_augmented if a["procedure"] == proc])
        print(f"  {proc}: {n}")


if __name__ == "__main__":
    main()
