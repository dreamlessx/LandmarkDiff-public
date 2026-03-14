"""Pre-flight check for training readiness.

Validates all components before submitting a training job:
1. Dataset exists and has correct structure
2. Model weights are cached and loadable
3. GPU is available and has enough VRAM
4. All imports work
5. A single forward pass succeeds

Usage:
    python scripts/preflight_check.py --data_dir data/training_combined --phase A
    python scripts/preflight_check.py --data_dir data/training_combined --phase B --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def check_dataset(data_dir: str) -> tuple[bool, dict]:
    """Check dataset structure and file counts."""
    d = Path(data_dir)
    info = {}

    if not d.exists():
        return False, {"error": f"Directory not found: {d}"}

    inputs = sorted(d.glob("*_input.png"))
    targets = sorted(d.glob("*_target.png"))
    conds = sorted(d.glob("*_conditioning.png"))
    masks = sorted(d.glob("*_mask.png"))

    info["input_files"] = len(inputs)
    info["target_files"] = len(targets)
    info["conditioning_files"] = len(conds)
    info["mask_files"] = len(masks)

    # Check consistency
    ok = True
    if len(inputs) == 0:
        ok = False
        info["error"] = "No input files found"
    if len(targets) < len(inputs):
        ok = False
        info["warning"] = f"Missing targets: {len(inputs) - len(targets)}"
    if len(conds) < len(inputs):
        ok = False
        info["warning"] = f"Missing conditioning: {len(inputs) - len(conds)}"

    # Check metadata
    meta_path = d / "metadata.json"
    info["has_metadata"] = meta_path.exists()
    if meta_path.exists():
        import json

        with open(meta_path) as f:
            meta = json.load(f)
        info["metadata_pairs"] = len(meta.get("pairs", {}))

    # Check a sample file
    if inputs:
        import cv2

        sample = cv2.imread(str(inputs[0]))
        if sample is not None:
            info["sample_shape"] = list(sample.shape)
        else:
            info["error"] = f"Cannot read sample: {inputs[0]}"
            ok = False

    return ok, info


def check_imports() -> tuple[bool, list[str]]:
    """Check all required imports."""
    failures = []
    modules = [
        "torch",
        "diffusers",
        "transformers",
        "cv2",
        "numpy",
        "PIL",
        "lpips",
        "mediapipe",
        "landmarkdiff.landmarks",
        "landmarkdiff.conditioning",
        "landmarkdiff.losses",
        "landmarkdiff.evaluation",
        "landmarkdiff.curriculum",
        "landmarkdiff.experiment_tracker",
        "landmarkdiff.augmentation",
        "landmarkdiff.validation",
        "landmarkdiff.arcface_torch",
    ]
    for mod in modules:
        try:
            __import__(mod)
        except ImportError:
            failures.append(mod)
    return len(failures) == 0, failures


def check_gpu() -> tuple[bool, dict]:
    """Check GPU availability and VRAM."""
    import torch

    info = {}
    if not torch.cuda.is_available():
        info["cuda"] = False
        info["device"] = "cpu"
        return False, info

    info["cuda"] = True
    info["device_count"] = torch.cuda.device_count()
    info["device_name"] = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    info["vram_total_gb"] = round(total, 1)
    info["bf16_supported"] = torch.cuda.is_bf16_supported()

    # Minimum 16GB VRAM for Phase A, 24GB recommended for Phase B
    ok = total >= 16.0
    if total < 24.0:
        info["warning"] = "Less than 24GB VRAM — Phase B may be tight"

    return ok, info


def check_model_cache() -> tuple[bool, dict]:
    """Check if SD1.5 and ControlNet weights are cached."""
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    info = {}

    sd15_dirs = list(cache_dir.glob("models--runwayml--stable-diffusion-v1-5*"))
    cn_dirs = list(cache_dir.glob("models--CrucibleAI--ControlNetMediaPipeFace*"))

    info["sd15_cached"] = len(sd15_dirs) > 0
    info["controlnet_cached"] = len(cn_dirs) > 0

    ok = info["sd15_cached"] and info["controlnet_cached"]
    if not ok:
        info["warning"] = "Missing models will be downloaded (~5GB)"

    return ok, info


def check_forward_pass(data_dir: str, phase: str = "A") -> tuple[bool, dict]:
    """Run a single forward pass to verify the training loop."""
    info = {}

    try:
        from scripts.train_controlnet import SyntheticPairDataset

        dataset = SyntheticPairDataset(data_dir, resolution=512, geometric_augment=False)
        sample = dataset[0]
        info["sample_keys"] = list(sample.keys())
        info["input_shape"] = list(sample["input"].shape)
        info["dtype"] = str(sample["input"].dtype)

        # Verify shapes
        assert sample["input"].shape == (3, 512, 512), f"Bad input shape: {sample['input'].shape}"
        assert sample["target"].shape == (3, 512, 512), (
            f"Bad target shape: {sample['target'].shape}"
        )
        assert sample["conditioning"].shape == (3, 512, 512), "Bad conditioning shape"

        info["shapes_valid"] = True
        return True, info
    except Exception as e:
        info["error"] = str(e)
        return False, info


def run_preflight(data_dir: str, phase: str = "A", quick: bool = False) -> bool:
    """Run all pre-flight checks."""
    print("=" * 60)
    print(f"  LandmarkDiff Pre-flight Check (Phase {phase})")
    print("=" * 60)
    print()

    all_ok = True

    # 1. Dataset
    print("1. Dataset check...")
    ok, info = check_dataset(data_dir)
    all_ok &= ok
    print(f"   [{check_mark(ok)}] {info.get('input_files', 0)} training pairs")
    if info.get("has_metadata"):
        print(f"   [{check_mark(True)}] Metadata found ({info.get('metadata_pairs', 0)} entries)")
    for key in ("error", "warning"):
        if key in info:
            print(f"   {'!!!' if key == 'error' else 'WRN'}: {info[key]}")
    print()

    # 2. Imports
    print("2. Import check...")
    ok, failures = check_imports()
    all_ok &= ok
    if ok:
        print(f"   [{check_mark(True)}] All modules importable")
    else:
        print(f"   [{check_mark(False)}] Missing: {', '.join(failures)}")
    print()

    # 3. GPU
    print("3. GPU check...")
    ok, info = check_gpu()
    if info.get("cuda"):
        print(f"   [{check_mark(ok)}] {info.get('device_name')} ({info.get('vram_total_gb')}GB)")
        print(f"   [{check_mark(info.get('bf16_supported', False))}] BF16 support")
    else:
        print(f"   [{check_mark(False)}] No CUDA device found")
        all_ok = False
    if "warning" in info:
        print(f"   WRN: {info['warning']}")
    print()

    # 4. Model cache
    print("4. Model cache check...")
    ok, info = check_model_cache()
    print(f"   [{check_mark(info.get('sd15_cached', False))}] SD1.5 weights")
    print(f"   [{check_mark(info.get('controlnet_cached', False))}] ControlNet weights")
    if "warning" in info:
        print(f"   WRN: {info['warning']}")
    print()

    # 5. Forward pass (skip if --quick)
    if not quick:
        print("5. Forward pass check...")
        ok, info = check_forward_pass(data_dir, phase)
        all_ok &= ok
        if ok:
            print(f"   [{check_mark(True)}] Dataset loading works")
            print(f"   [{check_mark(True)}] Sample shapes correct: {info.get('input_shape')}")
        else:
            print(f"   [{check_mark(False)}] {info.get('error', 'Unknown error')}")
    else:
        print("5. Forward pass check... SKIPPED (--quick)")
    print()

    # Summary
    print("=" * 60)
    if all_ok:
        print("  ALL CHECKS PASSED — Ready to train!")
    else:
        print("  SOME CHECKS FAILED — Fix issues before training")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-flight training check")
    parser.add_argument("--data_dir", default="data/training_combined")
    parser.add_argument("--phase", default="A", choices=["A", "B"])
    parser.add_argument("--quick", action="store_true", help="Skip forward pass check")
    args = parser.parse_args()

    ok = run_preflight(args.data_dir, args.phase, args.quick)
    sys.exit(0 if ok else 1)
