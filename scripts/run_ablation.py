"""Run ablation studies for paper Table 2.

Evaluates different loss combinations by loading Phase A/B checkpoints and
measuring their effect on each metric. Generates Table 2 (ablation) LaTeX.

Ablation configurations:
  1. L_diffusion only (Phase A baseline)
  2. L_diffusion + L_identity
  3. L_diffusion + L_perceptual
  4. L_diffusion + L_identity + L_perceptual (full Phase B)

Usage:
    # Evaluate existing checkpoints
    python scripts/run_ablation.py \
        --checkpoints_dir checkpoints_ablation/ \
        --test_dir data/test_pairs \
        --output results/ablation.json

    # Generate sbatch scripts for all ablation configs
    python scripts/run_ablation.py --generate_scripts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


ABLATION_CONFIGS = {
    "diffusion_only": {
        "label": r"$\mathcal{L}_\text{diff}$",
        "phase": "A",
        "description": "Diffusion loss only (Phase A baseline)",
    },
    "diff_identity": {
        "label": r"$\mathcal{L}_\text{diff} + \mathcal{L}_\text{id}$",
        "phase": "B",
        "loss_weights": {"identity": 0.05, "perceptual": 0.0},
        "description": "Diffusion + ArcFace identity",
    },
    "diff_perceptual": {
        "label": r"$\mathcal{L}_\text{diff} + \mathcal{L}_\text{perc}$",
        "phase": "B",
        "loss_weights": {"identity": 0.0, "perceptual": 0.1},
        "description": "Diffusion + LPIPS perceptual",
    },
    "full": {
        "label": r"$\mathcal{L}_\text{diff} + \mathcal{L}_\text{id} + \mathcal{L}_\text{perc}$",
        "phase": "B",
        "loss_weights": {"identity": 0.05, "perceptual": 0.1},
        "description": "Full 4-term loss (Phase B)",
    },
}


def generate_slurm_scripts(output_dir: str = "scripts/ablation") -> None:
    """Generate SLURM scripts for each ablation configuration."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, cfg in ABLATION_CONFIGS.items():
        phase = cfg["phase"]

        script = f"""#!/bin/bash
#SBATCH --job-name=abl_{name}
#SBATCH --partition=batch_gpu
#SBATCH --account=${{SLURM_ACCOUNT:-batch}}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-ablation-{name}-%j.out

set -euo pipefail
cd "${{LANDMARKDIFF_ROOT:-$(dirname $(dirname $(realpath $0)))}}"
source "${{CONDA_PREFIX:-$HOME/miniconda3}}"/etc/profile.d/conda.sh
conda activate ${{LANDMARKDIFF_ENV:-landmarkdiff}}

echo "Ablation: {name} — {cfg["description"]}"
echo "Start: $(date)"

python scripts/train_controlnet.py \\
    --data_dir data/training_combined \\
    --output_dir checkpoints_ablation/{name} \\
    --num_train_steps 30000 \\
    --train_batch_size {"4" if phase == "A" else "2"} \\
    --gradient_accumulation_steps {"4" if phase == "A" else "8"} \\
    --learning_rate {"1e-5" if phase == "A" else "5e-6"} \\
    --checkpoint_every 10000 \\
    --sample_every 5000 \\
    --log_every 100 \\
    --ema_decay 0.9999 \\
    --phase {phase} \\
    {"--resume_phaseA checkpoints_phaseA/final" if phase == "B" else ""} \\
    --seed 42

echo "Done: $(date)"
"""
        script_path = out / f"ablation_{name}.sh"
        with open(script_path, "w") as f:
            f.write(script)
        print(f"Generated: {script_path}")

    # Master submission script
    master = """#!/bin/bash
# Submit all ablation experiments
# Usage: bash scripts/ablation/submit_all.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting ablation experiments..."
"""
    for name in ABLATION_CONFIGS:
        master += f'JOB_{name.upper()}=$(sbatch --parsable "$SCRIPT_DIR/ablation_{name}.sh")\n'
        master += f'echo "  {name}: $JOB_{name.upper()}"\n'

    master += '\necho "All ablation jobs submitted."'

    with open(out / "submit_all.sh", "w") as f:
        f.write(master)
    print(f"\nMaster script: {out / 'submit_all.sh'}")


def evaluate_ablation(
    checkpoints_dir: str,
    test_dir: str,
    output_path: str,
    num_inference_steps: int = 20,
    max_samples: int = 200,
) -> dict:
    """Evaluate all ablation checkpoints and produce comparison JSON."""
    import tempfile
    import time

    import cv2
    import numpy as np
    import torch

    from landmarkdiff.evaluation import (
        compute_fid,
        compute_identity_similarity,
        compute_lpips,
        compute_ssim,
    )
    from landmarkdiff.landmarks import extract_landmarks

    ckpt_base = Path(checkpoints_dir)
    test_path = Path(test_dir)

    input_files = sorted(test_path.glob("*_input.png"))
    if max_samples > 0:
        input_files = input_files[:max_samples]

    if not input_files:
        print(f"No test pairs in {test_dir}")
        return {}

    results = {}

    for name, cfg in ABLATION_CONFIGS.items():
        ckpt_path = ckpt_base / name / "final" / "controlnet_ema"
        if not ckpt_path.exists():
            ckpt_path = ckpt_base / name / "final"
        if not ckpt_path.exists():
            # Try latest checkpoint
            candidates = sorted((ckpt_base / name).glob("checkpoint-*"))
            if candidates:
                ckpt_path = candidates[-1] / "controlnet_ema"

        if not ckpt_path.exists():
            print(f"  {name}: checkpoint not found, skipping")
            results[name] = {"status": "missing"}
            continue

        print(f"\nEvaluating {name}: {cfg['description']}")
        print(f"  Checkpoint: {ckpt_path}")

        # Load pipeline
        from landmarkdiff.inference import LandmarkDiffPipeline

        pipe = LandmarkDiffPipeline(
            mode="controlnet",
            controlnet_checkpoint=str(ckpt_path),
        )
        pipe.load()

        ssim_vals = []
        lpips_vals = []
        id_vals = []
        nme_vals = []
        fid_gen_dir = Path(tempfile.mkdtemp(prefix=f"abl_{name}_gen_"))
        fid_real_dir = Path(tempfile.mkdtemp(prefix=f"abl_{name}_real_"))

        t0 = time.time()
        for i, inp_file in enumerate(input_files):
            prefix = inp_file.stem.replace("_input", "")
            target_file = test_path / f"{prefix}_target.png"
            if not target_file.exists():
                continue

            input_img = cv2.imread(str(inp_file))
            target_img = cv2.imread(str(target_file))
            if input_img is None or target_img is None:
                continue

            input_img = cv2.resize(input_img, (512, 512))
            target_img = cv2.resize(target_img, (512, 512))

            # Infer procedure
            procedure = "rhinoplasty"
            for proc in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
                if proc in prefix:
                    procedure = proc
                    break

            try:
                result = pipe.generate(
                    input_img,
                    procedure=procedure,
                    num_inference_steps=num_inference_steps,
                    seed=42,
                )
                pred_img = result["output"]
            except Exception as e:
                print(f"    Skip {prefix}: {e}")
                continue

            # Metrics
            ssim_vals.append(compute_ssim(pred_img, target_img))
            lpips_val = compute_lpips(pred_img, target_img)
            if lpips_val is not None:
                lpips_vals.append(lpips_val)
            id_vals.append(compute_identity_similarity(pred_img, target_img))

            # NME
            pred_face = extract_landmarks(pred_img)
            target_face = extract_landmarks(target_img)
            if pred_face is not None and target_face is not None:
                from landmarkdiff.evaluation import compute_nme

                nme_val = compute_nme(
                    pred_face.pixel_coords,
                    target_face.pixel_coords,
                )
                nme_vals.append(nme_val)

            # Save for FID
            cv2.imwrite(str(fid_gen_dir / f"{i:06d}.png"), pred_img)
            cv2.imwrite(str(fid_real_dir / f"{i:06d}.png"), target_img)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f"    [{i + 1}/{len(input_files)}] "
                    f"SSIM={np.mean(ssim_vals):.4f} "
                    f"LPIPS={np.mean(lpips_vals):.4f} "
                    f"({elapsed:.0f}s)"
                )

        # FID
        try:
            fid = compute_fid(str(fid_real_dir), str(fid_gen_dir))
        except Exception:
            fid = None

        results[name] = {
            "label": cfg["label"],
            "description": cfg["description"],
            "metrics": {
                "ssim": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
                "lpips": float(np.mean(lpips_vals)) if lpips_vals else 0.0,
                "nme": float(np.mean(nme_vals)) if nme_vals else 0.0,
                "identity_sim": float(np.mean(id_vals)) if id_vals else 0.0,
                "fid": fid,
            },
            "num_samples": len(ssim_vals),
        }

        # Cleanup
        import shutil

        shutil.rmtree(fid_gen_dir, ignore_errors=True)
        shutil.rmtree(fid_real_dir, ignore_errors=True)

        # Unload model
        del pipe
        torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'=' * 70}")
    print("ABLATION RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Config':<30} {'SSIM':>8} {'LPIPS':>8} {'NME':>8} {'ID Sim':>8} {'FID':>8}")
    print("-" * 70)
    for name, r in results.items():
        if "metrics" not in r:
            print(f"{name:<30} {'MISSING':>8}")
            continue
        m = r["metrics"]
        fid_str = f"{m['fid']:.1f}" if m["fid"] is not None else "--"
        print(
            f"{name:<30} {m['ssim']:>8.4f} {m['lpips']:>8.4f} "
            f"{m['nme']:>8.4f} {m['identity_sim']:>8.4f} {fid_str:>8}"
        )

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def generate_ablation_latex(results: dict, output_path: str | None = None) -> str:
    """Generate LaTeX table for ablation study (Table 2 in paper)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study: effect of each loss component.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Loss Configuration & FID$\downarrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & NME$\downarrow$ & ArcFace$\uparrow$ \\",
        r"\midrule",
    ]

    for name in ["diffusion_only", "diff_identity", "diff_perceptual", "full"]:
        r = results.get(name, {})
        if "metrics" not in r:
            continue
        m = r["metrics"]
        label = r.get("label", name)
        fid_str = f"{m['fid']:.1f}" if m.get("fid") is not None else "--"

        bold = name == "full"
        fmt = lambda v, f=".4f": (r"\textbf{" + f"{v:{f}}" + "}") if bold else f"{v:{f}}"

        lines.append(
            f"{label} & {fid_str} & {fmt(m['ssim'])} & {fmt(m['lpips'])} & "
            f"{fmt(m['nme'])} & {fmt(m['identity_sim'])} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    latex = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to {output_path}")

    return latex


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--generate_scripts",
        action="store_true",
        help="Generate SLURM scripts for ablation configs",
    )
    parser.add_argument(
        "--checkpoints_dir",
        default="checkpoints_ablation",
        help="Directory containing ablation checkpoints",
    )
    parser.add_argument("--test_dir", default="data/test_pairs", help="Test pairs directory")
    parser.add_argument("--output", default="results/ablation.json", help="Output JSON path")
    parser.add_argument("--latex", default="paper/ablation_table.tex", help="Output LaTeX path")
    parser.add_argument("--max_samples", type=int, default=200, help="Max test samples to evaluate")
    parser.add_argument("--num_steps", type=int, default=20, help="Inference steps per sample")
    args = parser.parse_args()

    if args.generate_scripts:
        generate_slurm_scripts()
    else:
        results = evaluate_ablation(
            args.checkpoints_dir,
            args.test_dir,
            args.output,
            args.num_steps,
            args.max_samples,
        )
        if results:
            generate_ablation_latex(results, args.latex)
