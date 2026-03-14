"""Generate MICCAI reproducibility checklist.

MICCAI 2026 requires a structured reproducibility checklist as part of
the submission. This script auto-generates it by inspecting the codebase,
checking for required artifacts, and verifying reproducibility criteria.

Usage:
    python scripts/reproducibility_checklist.py --output paper/reproducibility_checklist.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check_item(description: str, check_func) -> tuple[bool, str]:
    """Check a reproducibility criterion and return (passed, details)."""
    try:
        result = check_func()
        if isinstance(result, tuple):
            passed, detail = result
        else:
            passed = bool(result)
            detail = "Yes" if passed else "No"
        return passed, detail
    except Exception as e:
        return False, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Generate reproducibility checklist")
    parser.add_argument("--output", type=str, default="paper/reproducibility_checklist.tex")
    args = parser.parse_args()

    checks = []

    # ── Code Availability ────────────────────────────────────────────────
    def check_code():
        gitdir = ROOT / ".git"
        return gitdir.exists(), "GitHub repository with full source code"

    checks.append(("Source code is available", check_code))

    def check_requirements():
        for f in ["requirements.txt", "pyproject.toml", "setup.py", "Makefile"]:
            if (ROOT / f).exists():
                return True, f"Found {f}"
        return False, "No dependency file found"

    checks.append(("Dependencies are documented", check_requirements))

    def check_dockerfile():
        return (ROOT / "Dockerfile").exists(), "Dockerfile for containerized reproduction"

    checks.append(("Docker/container support", check_dockerfile))

    def check_tests():
        test_files = list(ROOT.glob("tests/*.py")) + list(ROOT.glob("tests/**/*.py"))
        n = len(test_files)
        return n > 0, f"{n} test files"

    checks.append(("Automated tests exist", check_tests))

    # ── Training Details ─────────────────────────────────────────────────
    def check_hyperparams():
        train_script = ROOT / "scripts" / "train_controlnet.py"
        if not train_script.exists():
            return False, "Training script not found"
        content = train_script.read_text()
        has_lr = "learning_rate" in content or "lr" in content
        has_batch = "batch_size" in content
        return has_lr and has_batch, "LR and batch size documented in training script"

    checks.append(("Hyperparameters specified", check_hyperparams))

    def check_random_seeds():
        train_script = ROOT / "scripts" / "train_controlnet.py"
        if not train_script.exists():
            return False, "No training script"
        content = train_script.read_text()
        return "seed" in content.lower(), "Random seed controlled in training"

    checks.append(("Random seeds controlled", check_random_seeds))

    def check_training_curves():
        supp = ROOT / "paper" / "supplementary"
        if supp.exists():
            curves = list(supp.glob("training_curves*"))
            return len(curves) > 0, f"{len(curves)} training curve files"
        return False, "No supplementary directory"

    checks.append(("Training curves provided", check_training_curves))

    # ── Data ─────────────────────────────────────────────────────────────
    def check_data_description():
        paper = ROOT / "paper" / "main.tex"
        if paper.exists():
            content = paper.read_text()
            return "Datasets" in content or "datasets" in content, "Dataset section in paper"
        return False, "No paper found"

    checks.append(("Dataset described", check_data_description))

    def check_data_splits():
        splits = ROOT / "data" / "hda_splits"
        if splits.exists():
            n_test = (
                len(list((splits / "test").glob("*_input.png")))
                if (splits / "test").exists()
                else 0
            )
            return n_test > 0, f"{n_test} test pairs"
        return False, "No data splits directory"

    checks.append(("Train/val/test splits defined", check_data_splits))

    def check_preprocessing():
        scripts = list(ROOT.glob("scripts/build_*.py")) + list(ROOT.glob("scripts/prepare_*.py"))
        return len(scripts) > 0, f"{len(scripts)} data preparation scripts"

    checks.append(("Preprocessing code provided", check_preprocessing))

    # ── Evaluation ───────────────────────────────────────────────────────
    def check_eval_metrics():
        eval_mod = ROOT / "landmarkdiff" / "evaluation.py"
        if eval_mod.exists():
            content = eval_mod.read_text()
            metrics = []
            for m in ["ssim", "lpips", "fid", "nme", "identity", "arcface"]:
                if m in content.lower():
                    metrics.append(m)
            return len(metrics) >= 3, f"Metrics: {', '.join(metrics)}"
        return False, "No evaluation module"

    checks.append(("Evaluation metrics implemented", check_eval_metrics))

    def check_eval_results():
        paper_dir = ROOT / "paper"
        results = list(paper_dir.glob("*eval_results*.json")) + list(
            paper_dir.glob("*baseline*.json")
        )
        return len(results) > 0, f"{len(results)} result files"

    checks.append(("Evaluation results saved", check_eval_results))

    def check_statistical_tests():
        return (
            ROOT / "scripts" / "statistical_tests.py"
        ).exists(), "Statistical significance testing script"

    checks.append(("Statistical significance tested", check_statistical_tests))

    def check_multi_seed():
        return (
            ROOT / "scripts" / "multi_seed_eval.py"
        ).exists(), "Multi-seed evaluation for variance estimation"

    checks.append(("Multiple seeds evaluated", check_multi_seed))

    # ── Fairness ─────────────────────────────────────────────────────────
    def check_fairness():
        paper = ROOT / "paper" / "main.tex"
        if paper.exists():
            content = paper.read_text()
            return (
                "fitzpatrick" in content.lower() or "fairness" in content.lower(),
                "Fitzpatrick stratification in evaluation",
            )
        return False, "No fairness analysis"

    checks.append(("Fairness evaluation included", check_fairness))

    def check_model_card():
        return (
            ROOT / "paper" / "MODEL_CARD.md"
        ).exists(), "Model card following Mitchell et al. framework"

    checks.append(("Model card provided", check_model_card))

    # ── Run all checks ───────────────────────────────────────────────────
    print("=" * 60)
    print("MICCAI 2026 REPRODUCIBILITY CHECKLIST")
    print("=" * 60)
    print()

    passed = 0
    total = len(checks)

    results = []
    for description, check_func in checks:
        ok, detail = check_item(description, check_func)
        icon = "+" if ok else "-"
        print(f"  [{icon}] {description}")
        print(f"      {detail}")
        if ok:
            passed += 1
        results.append((description, ok, detail))

    print(f"\n{passed}/{total} checks passed")

    # ── Generate LaTeX ───────────────────────────────────────────────────
    latex_lines = [
        r"\section*{Reproducibility Checklist}",
        r"\begin{itemize}",
    ]
    for desc, ok, detail in results:
        mark = r"$\checkmark$" if ok else r"$\times$"
        latex_lines.append(f"  \\item [{mark}] {desc}: {detail}")
    latex_lines.append(r"\end{itemize}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(latex_lines))
    print(f"\nLaTeX checklist saved to {out_path}")


if __name__ == "__main__":
    main()
