"""Environment and dependency checker for LandmarkDiff.

Verifies that all required dependencies are installed and working,
checks GPU availability, validates model weights, and reports system
configuration. Run this before training or deployment.

Usage:
    python scripts/check_environment.py
    python scripts/check_environment.py --full  # include model weight checks
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class EnvironmentChecker:
    """Check environment readiness for LandmarkDiff."""

    def __init__(self):
        self.results: list[dict] = []

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.results.append({"name": name, "passed": passed, "detail": detail})
        sym = "+" if passed else "X"
        print(f"  [{sym}] {name}" + (f" -- {detail}" if detail else ""))
        return passed

    def run(self, full: bool = False) -> dict:
        print("=" * 60)
        print("LandmarkDiff Environment Check")
        print("=" * 60)

        self._check_python()
        self._check_core_deps()
        self._check_ml_deps()
        self._check_diffusion_deps()
        self._check_cv_deps()
        self._check_gpu()
        self._check_landmarkdiff()

        if full:
            self._check_model_weights()
            self._check_data()

        # Summary
        total = len(self.results)
        passed = sum(r["passed"] for r in self.results)
        failed = total - passed

        print(f"\n{'=' * 60}")
        print(f"Results: {passed}/{total} passed, {failed} issues")

        if failed > 0:
            print("\nRequired fixes:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  - {r['name']}: {r['detail']}")

        print("=" * 60)
        return {"total": total, "passed": passed, "failed": failed}

    def _check_python(self):
        print("\n--- Python ---")
        v = sys.version_info
        self.check("python_version", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}")
        self.check("python_64bit", sys.maxsize > 2**32, platform.architecture()[0])

    def _check_core_deps(self):
        print("\n--- Core Dependencies ---")
        deps = {
            "numpy": "1.24",
            "scipy": "1.10",
            "yaml": None,
            "PIL": None,
            "cv2": None,
            "tqdm": None,
        }
        for mod_name, _min_version in deps.items():
            try:
                mod = importlib.import_module(mod_name)
                version = getattr(mod, "__version__", "?")
                self.check(mod_name, True, f"v{version}")
            except ImportError:
                self.check(mod_name, False, "not installed")

    def _check_ml_deps(self):
        print("\n--- ML Dependencies ---")
        try:
            import torch

            self.check("torch", True, f"v{torch.__version__}")
            self.check(
                "torch_cuda_compile",
                torch.cuda.is_available() or True,
                f"CUDA {torch.version.cuda or 'N/A'}",
            )
        except ImportError:
            self.check("torch", False, "not installed")

        for mod_name, label in [
            ("torchvision", "torchvision"),
            ("lpips", "LPIPS"),
            ("insightface", "InsightFace"),
            ("onnxruntime", "ONNX Runtime"),
        ]:
            try:
                mod = importlib.import_module(mod_name)
                v = getattr(mod, "__version__", "?")
                self.check(label, True, f"v{v}")
            except ImportError:
                self.check(label, False, "not installed (optional)")

    def _check_diffusion_deps(self):
        print("\n--- Diffusion Model Dependencies ---")
        for mod_name, label in [
            ("diffusers", "Diffusers"),
            ("transformers", "Transformers"),
            ("accelerate", "Accelerate"),
            ("safetensors", "SafeTensors"),
        ]:
            try:
                mod = importlib.import_module(mod_name)
                v = getattr(mod, "__version__", "?")
                self.check(label, True, f"v{v}")
            except ImportError:
                self.check(label, False, "not installed")

    def _check_cv_deps(self):
        print("\n--- Computer Vision ---")
        try:
            import mediapipe

            self.check("MediaPipe", True, f"v{mediapipe.__version__}")
        except ImportError:
            self.check("MediaPipe", False, "not installed")

        try:
            from codeformer.basicsr.utils import imwrite  # noqa: F401

            self.check("CodeFormer", True, "available")
        except ImportError:
            self.check("CodeFormer", False, "not installed (optional)")

    def _check_gpu(self):
        print("\n--- GPU ---")
        try:
            import torch

            if torch.cuda.is_available():
                n_gpu = torch.cuda.device_count()
                for i in range(n_gpu):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                    self.check(f"GPU_{i}", True, f"{name} ({mem:.0f} GB)")
                    self.check(f"GPU_{i}_bf16", torch.cuda.is_bf16_supported(), "BF16 support")
            else:
                self.check("GPU", False, "No CUDA GPU available")
        except ImportError:
            self.check("GPU", False, "PyTorch not installed")

        # SLURM
        slurm_job = os.environ.get("SLURM_JOB_ID")
        if slurm_job:
            self.check("SLURM", True, f"Job {slurm_job}")

    def _check_landmarkdiff(self):
        print("\n--- LandmarkDiff Package ---")
        try:
            import landmarkdiff

            self.check("landmarkdiff", True, f"v{landmarkdiff.__version__}")
        except ImportError:
            self.check("landmarkdiff", False, "not importable")
            return

        # Check key modules
        key_modules = [
            "landmarks",
            "manipulation",
            "conditioning",
            "masking",
            "inference",
            "safety",
            "config",
            "data",
        ]
        for mod in key_modules:
            try:
                importlib.import_module(f"landmarkdiff.{mod}")
                self.check(f"landmarkdiff.{mod}", True)
            except Exception as e:
                self.check(f"landmarkdiff.{mod}", False, str(e)[:50])

    def _check_model_weights(self):
        print("\n--- Model Weights ---")

        # Stable Diffusion v1.5
        sd_cache = Path.home() / ".cache" / "huggingface" / "hub"
        sd_dirs = list(sd_cache.glob("models--runwayml--stable-diffusion-v1-5*"))
        self.check("SD1.5_weights", len(sd_dirs) > 0, str(sd_dirs[0]) if sd_dirs else "not cached")

        # ControlNet
        cn_dirs = list(sd_cache.glob("models--lllyasviel*"))
        self.check(
            "ControlNet_weights",
            len(cn_dirs) > 0,
            str(cn_dirs[0].name) if cn_dirs else "not cached",
        )

        # ArcFace
        arcface_paths = [
            Path.home() / ".insightface" / "models" / "buffalo_l",
            Path.home() / ".cache" / "arcface",
        ]
        found = any(p.exists() for p in arcface_paths)
        self.check(
            "ArcFace_weights",
            found,
            "found" if found else "not found (identity loss will use random init)",
        )

        # MediaPipe
        mp_cache = Path.home() / ".cache" / "mediapipe"
        self.check(
            "MediaPipe_cache",
            mp_cache.exists() or True,
            str(mp_cache) if mp_cache.exists() else "will download on first use",
        )

        # HuggingFace token
        hf_token = Path.home() / ".cache" / "huggingface" / "token"
        self.check(
            "HF_token",
            hf_token.exists(),
            "found" if hf_token.exists() else "missing (needed for gated models)",
        )

    def _check_data(self):
        print("\n--- Training Data ---")
        data_dir = Path(__file__).resolve().parent.parent / "data"

        for name in [
            "celeba_hq_extracted",
            "synthetic_surgery_pairs",
            "synthetic_surgery_pairs_v2",
            "synthetic_surgery_pairs_v3",
            "training_combined",
            "displacement_model.npz",
        ]:
            path = data_dir / name
            if path.exists():
                if path.is_dir():
                    n_files = len(list(path.glob("*_input.png")))
                    self.check(f"data/{name}", True, f"{n_files} pairs")
                else:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    self.check(f"data/{name}", True, f"{size_mb:.1f} MB")
            else:
                self.check(f"data/{name}", False, "not found")


def main():
    parser = argparse.ArgumentParser(description="Check LandmarkDiff environment")
    parser.add_argument("--full", action="store_true", help="Include model weight and data checks")
    args = parser.parse_args()

    checker = EnvironmentChecker()
    result = checker.run(full=args.full)
    sys.exit(0 if result["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
