"""Checkpoint quality checker for LandmarkDiff ControlNet training.

Usage:
    python scripts/check_checkpoint.py checkpoints_phaseA/checkpoint-5000/
    python scripts/check_checkpoint.py checkpoints_phaseA/checkpoint-5000/ --forward-pass
    python scripts/check_checkpoint.py checkpoints_phaseA/ --latest
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logger = logging.getLogger("check_checkpoint")


def find_latest_checkpoint(base_dir: Path) -> Path:
    """Find the checkpoint subdirectory with the highest step number."""
    candidates = sorted(
        (d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not candidates:
        logger.error("No checkpoint-* subdirectories found in %s", base_dir)
        sys.exit(1)
    logger.info("Auto-selected latest checkpoint: %s", candidates[-1].name)
    return candidates[-1]


def load_controlnet_state(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    """Load the ControlNet state dict from a diffusers checkpoint directory."""
    controlnet_dir = ckpt_dir / "controlnet"
    if not controlnet_dir.is_dir():
        logger.error("No controlnet/ subdirectory in %s", ckpt_dir)
        sys.exit(1)

    safetensors_files = list(controlnet_dir.glob("*.safetensors"))
    bin_files = list(controlnet_dir.glob("*.bin"))

    if safetensors_files:
        from safetensors.torch import load_file

        state_dict: dict[str, torch.Tensor] = {}
        for sf in safetensors_files:
            state_dict.update(load_file(str(sf), device="cpu"))
        return state_dict
    if bin_files:
        state_dict = {}
        for bf in bin_files:
            state_dict.update(torch.load(bf, map_location="cpu", weights_only=True))
        return state_dict
    logger.error("No .safetensors or .bin files in %s", controlnet_dir)
    sys.exit(1)


def check_key_coverage(state_dict: dict[str, torch.Tensor]) -> dict:
    """Check whether state dict keys match expected ControlNet architecture."""
    from diffusers import ControlNetModel

    ref_model = ControlNetModel.from_config(
        ControlNetModel.load_config("lllyasviel/control_v11p_sd15_openpose")
    )
    ref_keys = set(ref_model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = sorted(ref_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - ref_keys)
    return {
        "matched": len(ref_keys & ckpt_keys),
        "missing": missing,
        "unexpected": unexpected,
        "total_expected": len(ref_keys),
        "total_checkpoint": len(ckpt_keys),
        "pass": len(missing) == 0,
    }


def check_numeric_health(state_dict: dict[str, torch.Tensor]) -> dict:
    """Check all parameters for NaN and Inf values."""
    nan_keys: list[str] = []
    inf_keys: list[str] = []
    total_params = 0
    for key, tensor in state_dict.items():
        total_params += tensor.numel()
        if torch.isnan(tensor).any():
            nan_keys.append(key)
        if torch.isinf(tensor).any():
            inf_keys.append(key)
    return {
        "total_params": total_params,
        "total_params_human": f"{total_params / 1e6:.1f}M",
        "nan_keys": nan_keys,
        "inf_keys": inf_keys,
        "pass": len(nan_keys) == 0 and len(inf_keys) == 0,
    }


def check_metadata(ckpt_dir: Path) -> dict:
    """Read checkpoint metadata: step number, config, scheduler state."""
    controlnet_dir = ckpt_dir / "controlnet"
    result: dict = {"step": None, "config_found": False, "scheduler_found": False}
    name = ckpt_dir.name
    if name.startswith("checkpoint-"):
        with contextlib.suppress(ValueError):
            result["step"] = int(name.split("-")[-1])
    config_path = controlnet_dir / "config.json"
    if config_path.exists():
        result["config_found"] = True
        with open(config_path) as f:
            config = json.load(f)
        result["model_type"] = config.get("_class_name", "unknown")
        result["conditioning_channels"] = config.get("conditioning_channels", None)
    if (ckpt_dir / "scheduler.bin").exists():
        result["scheduler_found"] = True
    return result


def check_optimizer(ckpt_dir: Path) -> dict:
    """Verify optimizer state is present and loadable."""
    optimizer_path = ckpt_dir / "optimizer.bin"
    if not optimizer_path.exists():
        shards = list(ckpt_dir.glob("optimizer*.bin")) + list(
            ckpt_dir.glob("optimizer*.safetensors")
        )
        if not shards:
            return {"found": False, "valid": False, "detail": "No optimizer state files found"}
        return {
            "found": True,
            "valid": True,
            "shards": len(shards),
            "detail": f"{len(shards)} optimizer shard(s) found",
        }
    try:
        opt_state = torch.load(optimizer_path, map_location="cpu", weights_only=True)
        n_groups = len(opt_state.get("param_groups", []))
        return {
            "found": True,
            "valid": True,
            "param_groups": n_groups,
            "detail": f"Optimizer loaded, {n_groups} param group(s)",
        }
    except Exception as e:
        return {"found": True, "valid": False, "detail": f"Failed to load: {e}"}


def check_ema(ckpt_dir: Path) -> dict:
    """Check for EMA weights."""
    for candidate in [ckpt_dir / "ema_controlnet", ckpt_dir / "custom_checkpoint_0"]:
        if candidate.is_dir():
            files = list(candidate.glob("*.safetensors")) + list(candidate.glob("*.bin"))
            if files:
                return {"found": True, "location": candidate.name, "files": len(files)}
    if (ckpt_dir / "ema_weights.bin").exists():
        return {"found": True, "location": "ema_weights.bin"}
    return {"found": False, "detail": "No EMA weights detected (check use_ema config)"}


def run_forward_pass(ckpt_dir: Path) -> dict:
    """Load checkpoint and run a forward pass with dummy inputs."""
    from diffusers import ControlNetModel

    try:
        model = ControlNetModel.from_pretrained(
            str(ckpt_dir / "controlnet"), torch_dtype=torch.float32
        )
        model.eval()
        h, w = 8, 8
        with torch.no_grad():
            output = model(
                sample=torch.randn(1, 4, h, w),
                timestep=torch.tensor([500]),
                encoder_hidden_states=torch.randn(1, 77, 768),
                controlnet_cond=torch.randn(1, 3, h * 8, w * 8),
                return_dict=True,
            )
        down = output.down_block_res_samples
        mid = output.mid_block_res_sample
        finite = all(torch.isfinite(r).all() for r in down) and torch.isfinite(mid).all()
        return {
            "pass": finite,
            "n_down_blocks": len(down),
            "mid_block_shape": list(mid.shape),
            "output_finite": finite,
        }
    except Exception as e:
        return {"pass": False, "error": str(e)}


def build_report(ckpt_dir: Path, *, do_forward_pass: bool = False) -> dict:
    """Build the full checkpoint quality report."""
    report: dict = {"checkpoint_dir": str(ckpt_dir), "checks": {}}
    logger.info("Loading state dict from %s", ckpt_dir)
    state_dict = load_controlnet_state(ckpt_dir)

    logger.info("Checking numeric health...")
    report["checks"]["numeric_health"] = check_numeric_health(state_dict)
    logger.info("Checking key coverage against reference ControlNet...")
    report["checks"]["key_coverage"] = check_key_coverage(state_dict)
    logger.info("Reading metadata...")
    report["checks"]["metadata"] = check_metadata(ckpt_dir)
    logger.info("Checking optimizer state...")
    report["checks"]["optimizer"] = check_optimizer(ckpt_dir)
    logger.info("Checking EMA weights...")
    report["checks"]["ema"] = check_ema(ckpt_dir)

    if do_forward_pass:
        logger.info("Running forward pass with dummy inputs...")
        report["checks"]["forward_pass"] = run_forward_pass(ckpt_dir)

    critical = ["numeric_health", "key_coverage"]
    report["overall_pass"] = all(report["checks"][c].get("pass", False) for c in critical)
    return report


def print_summary(report: dict) -> None:
    """Print a human-readable summary to stderr via logging."""
    c = report["checks"]
    meta, nh, kc = c["metadata"], c["numeric_health"], c["key_coverage"]
    opt, ema = c["optimizer"], c["ema"]

    lines = ["=" * 60, f"Checkpoint Quality Report: {report['checkpoint_dir']}", "=" * 60]
    if meta["step"] is not None:
        lines.append(f"  Step: {meta['step']}")
    if meta.get("model_type"):
        lines.append(f"  Model: {meta['model_type']}")
    if meta.get("conditioning_channels") is not None:
        lines.append(f"  Conditioning channels: {meta['conditioning_channels']}")

    s = "PASS" if nh["pass"] else "FAIL"
    lines.append(f"  Parameters: {nh['total_params_human']}  [{s}]")
    if nh["nan_keys"]:
        lines.append(f"    NaN in {len(nh['nan_keys'])} tensor(s)")
    if nh["inf_keys"]:
        lines.append(f"    Inf in {len(nh['inf_keys'])} tensor(s)")

    s = "PASS" if kc["pass"] else "FAIL"
    lines.append(f"  Key coverage: {kc['matched']}/{kc['total_expected']}  [{s}]")
    if kc["missing"]:
        lines.append(f"    Missing: {len(kc['missing'])} key(s)")
    if kc["unexpected"]:
        lines.append(f"    Unexpected: {len(kc['unexpected'])} key(s)")

    lines.append(f"  Optimizer: {opt['detail']}")
    if ema["found"]:
        lines.append(f"  EMA: found at {ema.get('location', 'unknown')}")
    else:
        lines.append(f"  EMA: {ema.get('detail', 'not found')}")

    if "forward_pass" in c:
        fp = c["forward_pass"]
        s = "PASS" if fp.get("pass") else "FAIL"
        lines.append(f"  Forward pass: [{s}]")
        if fp.get("error"):
            lines.append(f"    Error: {fp['error']}")

    verdict = "PASS" if report["overall_pass"] else "FAIL"
    lines.extend(["=" * 60, f"  OVERALL: {verdict}", "=" * 60])
    for line in lines:
        logger.info(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check quality and integrity of a ControlNet training checkpoint.",
    )
    parser.add_argument("checkpoint_dir", type=Path, help="Path to checkpoint directory")
    parser.add_argument(
        "--latest", action="store_true", help="Auto-select the latest checkpoint-N subdirectory"
    )
    parser.add_argument(
        "--forward-pass",
        action="store_true",
        help="Run a forward pass with dummy inputs to verify model output",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    ckpt_dir: Path = args.checkpoint_dir.resolve()
    if not ckpt_dir.is_dir():
        logger.error("Directory does not exist: %s", ckpt_dir)
        sys.exit(1)
    if args.latest:
        ckpt_dir = find_latest_checkpoint(ckpt_dir)

    report = build_report(ckpt_dir, do_forward_pass=args.forward_pass)
    print_summary(report)
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
