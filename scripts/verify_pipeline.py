"""End-to-end pipeline integrity verification.

Runs each stage of the LandmarkDiff pipeline on a test image and verifies
shape consistency, value ranges, and determinism. Reports pass/fail for
each stage with detailed diagnostics.

Usage:
    python scripts/verify_pipeline.py [IMAGE]
    python scripts/verify_pipeline.py --synthetic  # generate test image
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class PipelineVerifier:
    """Verify each pipeline stage produces correct outputs."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: list[dict] = []
        self.artifacts: dict[str, object] = {}

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        self.results.append({"name": name, "passed": condition, "detail": detail})
        if self.verbose:
            sym = "+" if condition else "X"
            print(f"  [{sym}] {name}" + (f" — {detail}" if detail else ""))
        return condition

    def run(self, image: np.ndarray) -> dict:
        """Run full pipeline verification on an image."""
        print("=" * 60)
        print("LandmarkDiff Pipeline Integrity Check")
        print("=" * 60)

        self._verify_input(image)
        face = self._verify_landmarks(image)
        if face is not None:
            self._verify_manipulation(face)
            self._verify_conditioning(face, image)
            self._verify_masking(face)
            self._verify_tps_warp(image, face)
            self._verify_safety(image)
            self._verify_config()
            self._verify_determinism(image)

        # Summary
        total = len(self.results)
        passed = sum(r["passed"] for r in self.results)
        failed = total - passed

        print("\n" + "=" * 60)
        print(f"Results: {passed}/{total} passed, {failed} failed")
        if failed == 0:
            print("Pipeline integrity: ALL CHECKS PASSED")
        else:
            print("Pipeline integrity: SOME CHECKS FAILED")
            for r in self.results:
                if not r["passed"]:
                    print(f"  FAIL: {r['name']} — {r['detail']}")
        print("=" * 60)

        return {"total": total, "passed": passed, "failed": failed, "results": self.results}

    def _verify_input(self, image: np.ndarray) -> None:
        print("\n--- Stage 1: Input Validation ---")
        self.check("input_type", isinstance(image, np.ndarray))
        self.check("input_ndim", image.ndim == 3, f"ndim={image.ndim}")
        self.check("input_channels", image.shape[2] == 3, f"channels={image.shape[2]}")
        self.check("input_dtype", image.dtype == np.uint8, f"dtype={image.dtype}")
        self.check("input_min_size", min(image.shape[:2]) >= 64, f"shape={image.shape[:2]}")
        self.check(
            "input_value_range",
            image.min() >= 0 and image.max() <= 255,
            f"range=[{image.min()}, {image.max()}]",
        )

    def _verify_landmarks(self, image: np.ndarray):
        print("\n--- Stage 2: Landmark Extraction ---")
        try:
            from landmarkdiff.landmarks import extract_landmarks

            t0 = time.time()
            face = extract_landmarks(image)
            dt = time.time() - t0

            if face is None:
                self.check("landmarks_detected", False, "No face detected")
                return None

            self.check("landmarks_detected", True, f"confidence={face.confidence:.3f}")
            self.check(
                "landmarks_shape", face.landmarks.shape == (478, 3), f"shape={face.landmarks.shape}"
            )
            self.check(
                "landmarks_normalized",
                face.landmarks[:, :2].min() >= 0 and face.landmarks[:, :2].max() <= 1,
                f"range=[{face.landmarks[:, :2].min():.3f}, {face.landmarks[:, :2].max():.3f}]",
            )
            self.check("landmarks_speed", dt < 5.0, f"took {dt:.2f}s")

            self.artifacts["face"] = face
            return face
        except Exception as e:
            self.check("landmarks_import", False, str(e))
            return None

    def _verify_manipulation(self, face) -> None:
        print("\n--- Stage 3: Landmark Manipulation ---")
        try:
            from landmarkdiff.manipulation import apply_procedure_preset

            for proc in [
                "rhinoplasty",
                "blepharoplasty",
                "rhytidectomy",
                "orthognathic",
                "brow_lift",
                "mentoplasty",
            ]:
                result = apply_procedure_preset(face, proc, 50.0, image_size=512)
                manip_landmarks = result.landmarks
                self.check(
                    f"manip_{proc}_shape",
                    manip_landmarks.shape == face.landmarks.shape,
                    f"shape={manip_landmarks.shape}",
                )
                # Displaced landmarks should differ from original
                diff = np.abs(manip_landmarks - face.landmarks).sum()
                self.check(f"manip_{proc}_changed", diff > 0, f"total_diff={diff:.6f}")
        except Exception as e:
            self.check("manipulation_import", False, str(e))

    def _verify_conditioning(self, face, image: np.ndarray) -> None:
        print("\n--- Stage 4: Conditioning Generation ---")
        try:
            from landmarkdiff.conditioning import generate_conditioning

            cond = generate_conditioning(face, image.shape[1], image.shape[0])
            self.check(
                "cond_shape",
                cond.shape[:2] == image.shape[:2],
                f"cond={cond.shape}, img={image.shape}",
            )
            self.check("cond_channels", cond.shape[2] == 3, f"channels={cond.shape[2]}")
            self.check("cond_dtype", cond.dtype == np.uint8, f"dtype={cond.dtype}")
            self.check("cond_not_blank", cond.sum() > 0, "conditioning has content")
            self.artifacts["conditioning"] = cond
        except Exception as e:
            self.check("conditioning_import", False, str(e))

    def _verify_masking(self, face) -> None:
        print("\n--- Stage 5: Surgical Mask Generation ---")
        try:
            from landmarkdiff.masking import generate_surgical_mask

            for proc in [
                "rhinoplasty",
                "blepharoplasty",
                "rhytidectomy",
                "orthognathic",
                "brow_lift",
                "mentoplasty",
            ]:
                mask = generate_surgical_mask(face, proc)
                self.check(
                    f"mask_{proc}_shape",
                    mask.shape == (face.image_height, face.image_width),
                    f"shape={mask.shape}",
                )
                self.check(
                    f"mask_{proc}_range",
                    mask.min() >= 0 and mask.max() <= 1,
                    f"range=[{mask.min():.3f}, {mask.max():.3f}]",
                )
                self.check(f"mask_{proc}_not_blank", mask.sum() > 0)
            self.artifacts["mask"] = mask
        except Exception as e:
            self.check("masking_import", False, str(e))

    def _verify_tps_warp(self, image: np.ndarray, face) -> None:
        print("\n--- Stage 6: TPS Warp ---")
        try:
            from landmarkdiff.synthetic.tps_warp import warp_image_tps

            src = face.landmarks[:, :2].copy()
            dst = src + np.random.randn(*src.shape).astype(np.float32) * 0.003

            t0 = time.time()
            warped = warp_image_tps(image, src, dst)
            dt = time.time() - t0

            self.check(
                "tps_shape",
                warped.shape == image.shape,
                f"warped={warped.shape}, orig={image.shape}",
            )
            self.check("tps_dtype", warped.dtype == np.uint8)
            self.check(
                "tps_differs", not np.array_equal(warped, image), "warped differs from original"
            )
            self.check("tps_speed", dt < 10.0, f"took {dt:.2f}s")
            self.artifacts["warped"] = warped
        except Exception as e:
            self.check("tps_import", False, str(e))

    def _verify_safety(self, image: np.ndarray) -> None:
        print("\n--- Stage 7: Safety Validation ---")
        try:
            from landmarkdiff.safety import SafetyResult, SafetyValidator

            validator = SafetyValidator()
            output = (image.astype(float) * 0.95).astype(np.uint8)

            result = validator.validate(
                input_image=image,
                output_image=output,
                face_confidence=0.95,
            )
            self.check("safety_type", isinstance(result, SafetyResult))
            self.check(
                "safety_has_checks", len(result.checks) > 0, f"checks: {list(result.checks.keys())}"
            )

            # Watermark test
            watermarked = validator.apply_watermark(image)
            self.check("watermark_shape", watermarked.shape == image.shape)
            self.check("watermark_differs", not np.array_equal(watermarked, image))
        except Exception as e:
            self.check("safety_import", False, str(e))

    def _verify_config(self) -> None:
        print("\n--- Stage 8: Configuration ---")
        try:
            from landmarkdiff.config import load_config, validate_config

            cfg = load_config()
            self.check("config_defaults", cfg.experiment_name == "default")
            self.check("config_version", cfg.version == "0.3.0", f"version={cfg.version}")

            warnings = validate_config(cfg)
            self.check("config_valid", isinstance(warnings, list), f"warnings={len(warnings)}")
        except Exception as e:
            self.check("config_import", False, str(e))

    def _verify_determinism(self, image: np.ndarray) -> None:
        print("\n--- Stage 9: Determinism ---")
        try:
            from landmarkdiff.landmarks import extract_landmarks

            face1 = extract_landmarks(image)
            face2 = extract_landmarks(image)

            if face1 is not None and face2 is not None:
                diff = np.abs(face1.landmarks - face2.landmarks).max()
                self.check("deterministic_landmarks", diff < 1e-4, f"max_diff={diff:.6f}")
        except Exception as e:
            self.check("determinism_import", False, str(e))


def create_synthetic_test_image() -> np.ndarray:
    """Create a synthetic face-like test image."""
    img = np.full((512, 512, 3), 180, dtype=np.uint8)

    # Skin-tone base
    img[:, :, 0] = 150  # B
    img[:, :, 1] = 170  # G
    img[:, :, 2] = 200  # R (redder = skin-like)

    # Oval face shape
    cv2.ellipse(img, (256, 256), (140, 180), 0, 0, 360, (140, 160, 190), -1)

    # Eyes (dark)
    cv2.circle(img, (200, 220), 15, (50, 50, 50), -1)
    cv2.circle(img, (312, 220), 15, (50, 50, 50), -1)

    # Nose
    pts = np.array([[256, 250], [240, 300], [272, 300]], np.int32)
    cv2.fillPoly(img, [pts], (130, 150, 180))

    # Mouth
    cv2.ellipse(img, (256, 340), (40, 15), 0, 0, 360, (100, 120, 170), -1)

    # Add some noise for texture
    noise = np.random.default_rng(42).integers(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LandmarkDiff Pipeline Integrity Check")
    parser.add_argument("image", nargs="?", help="Input image path")
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic test image instead of a real one"
    )
    parser.add_argument("--output", default=None, help="Save artifacts to this directory")
    args = parser.parse_args()

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"ERROR: Cannot read image: {args.image}")
            sys.exit(1)
        image = cv2.resize(image, (512, 512))
    elif args.synthetic:
        image = create_synthetic_test_image()
        print("Using synthetic test image")
    else:
        # Try to find a test image in the data directory
        data_dir = Path(__file__).resolve().parent.parent / "data"
        test_images = list(data_dir.glob("**/celeba_hq_extracted/*.png"))[:1]
        if test_images:
            image = cv2.imread(str(test_images[0]))
            image = cv2.resize(image, (512, 512))
            print(f"Using: {test_images[0].name}")
        else:
            image = create_synthetic_test_image()
            print("No test images found, using synthetic")

    verifier = PipelineVerifier()
    result = verifier.run(image)

    # Save artifacts if requested
    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "input.png"), image)
        for name, artifact in verifier.artifacts.items():
            if isinstance(artifact, np.ndarray) and artifact.ndim >= 2:
                if artifact.dtype == np.float32:
                    artifact = (artifact * 255).astype(np.uint8)
                if artifact.ndim == 2:
                    artifact = cv2.cvtColor(artifact, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(out / f"{name}.png"), artifact)
        print(f"\nArtifacts saved to: {out}")

    sys.exit(0 if result["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
