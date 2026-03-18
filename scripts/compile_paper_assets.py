"""Compile all paper assets into a structured report.

Generates a summary of all available results, figures, and metrics
to help verify that the paper is consistent with the data. Also
checks that all figures referenced in main.tex exist and are non-empty.

Usage:
    python scripts/compile_paper_assets.py
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"


def check_latex_figures():
    """Check that all figures referenced in main.tex exist."""
    tex_path = PAPER / "main.tex"
    if not tex_path.exists():
        return []

    content = tex_path.read_text()
    issues = []

    # Find all \includegraphics references
    import re

    for match in re.finditer(r"\\includegraphics.*?\{([^}]+)\}", content):
        fig_name = match.group(1)
        fig_path = PAPER / fig_name
        if not fig_path.exists():
            issues.append(f"MISSING: {fig_name}")
        elif fig_path.stat().st_size < 100:
            issues.append(f"EMPTY: {fig_name} ({fig_path.stat().st_size} bytes)")
        else:
            size_kb = fig_path.stat().st_size / 1024
            issues.append(f"OK: {fig_name} ({size_kb:.0f} KB)")

    return issues


def check_latex_citations():
    """Check that all citations have bib entries."""
    tex_path = PAPER / "main.tex"
    bib_path = PAPER / "references.bib"
    if not tex_path.exists() or not bib_path.exists():
        return []

    import re

    tex = tex_path.read_text()
    bib = bib_path.read_text()

    # Extract all \cite{...} keys
    cite_keys = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", tex):
        for key in match.group(1).split(","):
            cite_keys.add(key.strip())

    # Extract all @type{key, entries
    bib_keys = set(re.findall(r"@\w+\{(\w+)", bib))

    issues = []
    for key in sorted(cite_keys):
        if key in bib_keys:
            issues.append(f"OK: {key}")
        else:
            issues.append(f"MISSING BIB: {key}")

    # Unused bib entries
    unused = bib_keys - cite_keys
    if unused:
        for key in sorted(unused):
            issues.append(f"UNUSED BIB: {key}")

    return issues


def summarize_json_results():
    """Summarize all JSON result files in paper/."""
    summaries = []
    for json_file in sorted(PAPER.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            summary = f"\n--- {json_file.name} ---"
            size_kb = json_file.stat().st_size / 1024

            if isinstance(data, dict):
                summary += f" ({size_kb:.1f} KB, {len(data)} top-level keys)"
                # Try to extract key metrics
                if "metrics" in data:
                    m = data["metrics"]
                    if isinstance(m, dict):
                        for k, v in sorted(m.items()):
                            if isinstance(v, dict) and "mean" in v:
                                summary += f"\n  {k}: {v['mean']:.4f}"
                            elif isinstance(v, int | float):
                                summary += f"\n  {k}: {v:.4f}"

                if "per_procedure" in data:
                    summary += "\n  Procedures: " + ", ".join(sorted(data["per_procedure"].keys()))

            summaries.append(summary)
        except Exception as e:
            summaries.append(f"\n--- {json_file.name} --- ERROR: {e}")

    return summaries


def count_codebase():
    """Count lines of code in the codebase."""
    counts = {}
    for pattern, label in [
        ("landmarkdiff/*.py", "Core modules"),
        ("scripts/*.py", "Scripts"),
        ("tests/*.py", "Tests"),
        ("slurm/*.sh", "SLURM scripts"),
    ]:
        files = list(ROOT.glob(pattern))
        total_lines = 0
        for f in files:
            with contextlib.suppress(Exception):
                total_lines += len(f.read_text().splitlines())
        counts[label] = (len(files), total_lines)
    return counts


def main():
    print("=" * 70)
    print("LANDMARKDIFF PAPER ASSETS COMPILATION")
    print("=" * 70)

    # 1. Figure check
    print("\n[1] FIGURE REFERENCES")
    for issue in check_latex_figures():
        print(f"  {issue}")

    # 2. Citation check
    print("\n[2] CITATION REFERENCES")
    citations = check_latex_citations()
    ok = sum(1 for c in citations if c.startswith("OK"))
    missing = sum(1 for c in citations if "MISSING" in c)
    unused = sum(1 for c in citations if "UNUSED" in c)
    print(f"  {ok} citations OK, {missing} missing, {unused} unused bib entries")
    for c in citations:
        if "MISSING" in c or "UNUSED" in c:
            print(f"  {c}")

    # 3. Result files
    print("\n[3] RESULT FILES")
    for s in summarize_json_results():
        print(f"  {s}")

    # 4. Paper figures
    print("\n[4] PAPER FIGURES")
    figs = list(PAPER.glob("fig_*.png")) + list(PAPER.glob("*.png"))
    figs = sorted(set(figs))
    total_size = 0
    for f in figs:
        size_kb = f.stat().st_size / 1024
        total_size += size_kb
        print(f"  {f.name}: {size_kb:.0f} KB")
    print(f"  Total: {len(figs)} figures, {total_size / 1024:.1f} MB")

    # 5. Supplementary materials
    print("\n[5] SUPPLEMENTARY MATERIALS")
    supp_dir = PAPER / "supplementary"
    if supp_dir.exists():
        for f in sorted(supp_dir.glob("*")):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name}: {size_kb:.0f} KB")
    else:
        print("  No supplementary directory")

    # 6. Attention maps
    print("\n[6] ATTENTION MAPS")
    attn_dir = PAPER / "attention_maps"
    if attn_dir.exists():
        n = len(list(attn_dir.glob("*.png")))
        total_mb = sum(f.stat().st_size for f in attn_dir.glob("*.png")) / 1e6
        print(f"  {n} attention map files, {total_mb:.1f} MB total")
    else:
        print("  No attention maps")

    # 7. Codebase stats
    print("\n[7] CODEBASE STATISTICS")
    counts = count_codebase()
    total_files = 0
    total_lines = 0
    for label, (n_files, n_lines) in sorted(counts.items()):
        print(f"  {label}: {n_files} files, {n_lines:,} lines")
        total_files += n_files
        total_lines += n_lines
    print(f"  Total: {total_files} files, {total_lines:,} lines")

    # 8. SLURM scripts
    print("\n[8] SLURM SCRIPTS")
    for f in sorted((ROOT / "slurm").glob("*.sh")):
        print(f"  {f.name}")

    print(f"\n{'=' * 70}")
    print("COMPILATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
