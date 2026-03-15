r"""Auto-update LaTeX tables in paper/main.tex with evaluation results.

Reads all available evaluation result JSON files (baseline metrics, SD1.5 img2img
baselines at various strengths, LandmarkDiff per-seed and aggregated results) and
replaces the placeholder tables in paper/main.tex with real numeric values.

Handles:
  - Table 1 (tab:results): per-procedure comparison across 5 methods x 5 metrics
  - Table 3 (tab:fitzpatrick): Fitzpatrick skin type stratification

Design decisions:
  - Missing files are skipped gracefully with a warning, and affected cells
    become "--" placeholders.
  - For multi-seed LandmarkDiff results, values are formatted as $mean{\pm}std$.
  - The best value per metric per procedure is bolded automatically.
  - SD baseline: the strength with the best overall SSIM is selected.
  - Fitzpatrick types I-II and V-VI are grouped together (weighted avg by n).

Usage:
    python scripts/update_paper_tables.py
    python scripts/update_paper_tables.py --paper paper/main.tex --dry-run

Author: Red agent
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (two levels up from scripts/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent


# ====================================================================
# Utility helpers
# ====================================================================


def load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None if it doesn't exist or is malformed."""
    if not path.exists():
        print(f"  [SKIP] {path.relative_to(ROOT)} not found")
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"  [OK]   {path.relative_to(ROOT)} loaded")
        return data
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] {path.relative_to(ROOT)} failed to load: {exc}")
        return None


def fmt_val(value: float | None, decimals: int) -> str:
    """Format a single numeric value to the specified decimal places.

    Returns "--" for None / NaN.  Uses Python string formatting to ensure
    consistent trailing zeros (e.g. 0.530, not 0.53).
    """
    if value is None:
        return "--"
    try:
        return f"{value:.{decimals}f}"
    except (TypeError, ValueError):
        return "--"


def fmt_pm(mean: float | None, std: float | None, decimals: int) -> str:
    r"""Format a mean +/- std pair as LaTeX: $0.533{\pm}0.012$.

    If std is None or zero, just returns the mean value without ±.
    """
    if mean is None:
        return "--"
    m = f"{mean:.{decimals}f}"
    if std is not None and std > 0:
        s = f"{std:.{decimals}f}"
        return f"${m}" + r"{\pm}" + f"{s}$"
    return m


def decimals_for_metric(metric: str) -> int:
    """Return the number of decimal places appropriate for each metric.

    - FID: 1 decimal place (e.g. 23.4)
    - LPIPS, NME: 3 decimal places (e.g. 0.378)
    - SSIM, ArcFace/identity_sim: 3 decimal places (e.g. 0.533)
    """
    metric_lower = metric.lower()
    if "fid" in metric_lower:
        return 1
    # Everything else: 3 decimal places
    return 3


def bold_best(
    values: list[str],
    higher_is_better: bool,
    raw_values: list[float | None],
) -> list[str]:
    """Bold the best value in a list of formatted strings.

    Determines the best by comparing raw numeric values (ignoring None).
    If higher_is_better, the max wins; otherwise the min wins.
    The formatted string for the winner gets wrapped in \\textbf{}.

    If a value already contains LaTeX math ($...{\\pm}...$), we bold the
    entire expression.  We also handle the case where the value is already
    wrapped in textbf (idempotent).
    """
    # Filter out unavailable entries ("--")
    valid_indices = [i for i, v in enumerate(raw_values) if v is not None]
    if not valid_indices:
        return values  # nothing to bold

    if higher_is_better:
        best_val = max(raw_values[i] for i in valid_indices)
    else:
        best_val = min(raw_values[i] for i in valid_indices)

    result = list(values)
    for i in valid_indices:
        if raw_values[i] == best_val:
            cell = result[i]
            # Don't double-bold
            if r"\textbf{" not in cell:
                result[i] = r"\textbf{" + cell + "}"
    return result


# ====================================================================
# Data extraction helpers
# ====================================================================

# The canonical order of procedures in Table 1
PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]

# Sample sizes per procedure (from baseline_results.json, also hardcoded in
# the LaTeX template).  Will be overridden from actual data if available.
DEFAULT_N = {
    "rhinoplasty": 21,
    "blepharoplasty": 27,
    "rhytidectomy": 9,
    "orthognathic": 10,
}

# Metrics in Table 1 column order
TABLE1_METRICS = ["fid", "lpips", "nme", "ssim", "identity_sim"]

# Whether higher is better for each metric
HIGHER_IS_BETTER = {
    "fid": False,
    "lpips": False,
    "nme": False,
    "ssim": True,
    "identity_sim": True,
}


def get_baseline_val(
    baseline_data: dict | None,
    procedure: str,
    prefix: str,
    metric: str,
) -> float | None:
    """Extract a metric value from the baseline_results.json structure.

    The baseline JSON has keys like:
        baseline_data[procedure][f"{prefix}_{metric}"]["mean"]

    where prefix is one of: direct, tps, morph
    and metric is one of: ssim, lpips, nme
    (No FID or identity_sim in baseline data.)
    """
    if baseline_data is None:
        return None
    proc = baseline_data.get(procedure)
    if proc is None:
        return None
    key = f"{prefix}_{metric}"
    entry = proc.get(key)
    if entry is None:
        return None
    return entry.get("mean")


def get_sd_val(
    sd_data: dict | None,
    procedure: str,
    metric: str,
) -> float | None:
    """Extract a metric value from an SD img2img baseline JSON.

    Structure: sd_data[procedure][metric]["mean"]
    """
    if sd_data is None:
        return None
    proc = sd_data.get(procedure)
    if proc is None:
        return None
    entry = proc.get(metric)
    if entry is None:
        return None
    return entry.get("mean")


def get_eval_val(
    eval_data: dict | None,
    procedure: str,
    metric: str,
) -> float | None:
    """Extract a metric value from a LandmarkDiff eval results JSON.

    Structure: eval_data[procedure][metric]["mean"]
    Same structure as SD baseline files.
    """
    # Same structure as SD baseline
    return get_sd_val(eval_data, procedure, metric)


def select_best_sd_strength(
    sd_files: dict[float, dict | None],
) -> tuple[float, dict | None]:
    """Pick the SD img2img strength with the best overall SSIM.

    Args:
        sd_files: mapping from strength (e.g. 0.3) to loaded JSON (or None).

    Returns:
        (best_strength, best_data) tuple.  If no files are available,
        returns (0.3, None) as a fallback.
    """
    best_strength = 0.3
    best_ssim = -1.0
    best_data = None

    for strength, data in sd_files.items():
        if data is None:
            continue
        overall = data.get("overall", {})
        ssim_entry = overall.get("ssim", {})
        ssim_mean = ssim_entry.get("mean")
        if ssim_mean is not None and ssim_mean > best_ssim:
            best_ssim = ssim_mean
            best_strength = strength
            best_data = data

    print(f"  [INFO] Best SD img2img strength: {best_strength} (overall SSIM={best_ssim:.4f})")
    return best_strength, best_data


def aggregate_seeds(
    seed_files: dict[int, dict | None],
) -> dict | None:
    """Aggregate multi-seed LandmarkDiff results into mean ± std.

    If an aggregated file already exists, prefer that.  Otherwise, compute
    from individual seed files.

    Returns a dict with the same procedure/metric structure, but each metric
    entry has both "mean" and "std" keys representing the cross-seed statistics.
    """
    import numpy as np

    # Collect all valid seed results
    valid = {s: d for s, d in seed_files.items() if d is not None}
    if not valid:
        return None

    # If only one seed, return it directly (std = 0)
    if len(valid) == 1:
        print("  [INFO] Only 1 seed available, std will be 0")
        data = next(iter(valid.values()))
        # Deep copy and set std to 0 for all metrics
        result = {}
        for proc in PROCEDURES + ["overall"]:
            if proc not in data:
                continue
            result[proc] = {}
            for metric in TABLE1_METRICS:
                if metric in data[proc]:
                    result[proc][metric] = {
                        "mean": data[proc][metric].get("mean"),
                        "std": 0.0,
                    }
        return result

    # Multiple seeds: compute cross-seed mean and std
    print(f"  [INFO] Aggregating {len(valid)} seeds: {sorted(valid.keys())}")
    result = {}
    for proc in PROCEDURES + ["overall"]:
        result[proc] = {}
        for metric in TABLE1_METRICS:
            vals = []
            for seed_data in valid.values():
                proc_data = seed_data.get(proc, {})
                metric_data = proc_data.get(metric, {})
                v = metric_data.get("mean")
                if v is not None:
                    vals.append(v)
            if vals:
                result[proc][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
    return result


# ====================================================================
# Fitzpatrick grouping helpers
# ====================================================================


def weighted_mean(values: list[tuple[float, int]]) -> float | None:
    """Compute a weighted mean from (value, n) pairs.

    Returns None if no valid entries.
    """
    total_n = sum(n for _, n in values if n > 0)
    if total_n == 0:
        return None
    return sum(v * n for v, n in values) / total_n


def group_fitzpatrick(
    fitz_data: dict | None,
) -> dict[str, dict]:
    """Group Fitzpatrick types into I-II, III, IV, V-VI.

    Returns a dict mapping group name -> {metric -> {"mean": ..., "n": ...}}.
    Groups I-II and V-VI are computed as weighted averages by sample size.
    """
    if fitz_data is None:
        return {}

    groups = {}
    metrics = ["ssim", "lpips", "nme", "identity_sim"]

    # Group I-II: weighted average of types I and II
    group_name = "I--II"
    groups[group_name] = {}
    for m in metrics:
        pairs = []  # (mean_val, n)
        total_n = 0
        for t in ["I", "II"]:
            t_data = fitz_data.get(t, {})
            m_data = t_data.get(m, {})
            val = m_data.get("mean")
            n = m_data.get("n", 0)
            if val is not None and n > 0:
                pairs.append((val, n))
                total_n += n
        groups[group_name][m] = {
            "mean": weighted_mean(pairs),
            "n": total_n,
        }

    # Groups III and IV: direct values
    for t in ["III", "IV"]:
        t_data = fitz_data.get(t, {})
        groups[t] = {}
        for m in metrics:
            m_data = t_data.get(m, {})
            groups[t][m] = {
                "mean": m_data.get("mean"),
                "n": m_data.get("n", 0),
            }

    # Group V-VI: weighted average
    group_name = "V--VI"
    groups[group_name] = {}
    for m in metrics:
        pairs = []
        total_n = 0
        for t in ["V", "VI"]:
            t_data = fitz_data.get(t, {})
            m_data = t_data.get(m, {})
            val = m_data.get("mean")
            n = m_data.get("n", 0)
            if val is not None and n > 0:
                pairs.append((val, n))
                total_n += n
        groups[group_name][m] = {
            "mean": weighted_mean(pairs),
            "n": total_n,
        }

    return groups


# ====================================================================
# Table generation
# ====================================================================


def build_table1_body(
    baseline_data: dict | None,
    sd_data: dict | None,
    ld_agg: dict | None,
    has_multi_seed: bool,
) -> str:
    """Build the body of Table 1 (tab:results).

    For each procedure, generates rows for 5 methods (Direct copy, TPS-only,
    Morphing, SD1.5 Img2Img, LandmarkDiff) and 5 metrics (FID, LPIPS, NME,
    SSIM, ArcFace).

    The best value per metric per procedure is bolded.  LandmarkDiff values
    are shown as mean +/- std when multi-seed results are available.
    """
    lines = []

    for proc in PROCEDURES:
        # Determine sample size (from baseline data or default)
        n = DEFAULT_N.get(proc, 0)
        if baseline_data and proc in baseline_data:
            # Try to get n from any metric entry
            for key in baseline_data[proc]:
                entry = baseline_data[proc][key]
                if isinstance(entry, dict) and "n" in entry:
                    n = entry["n"]
                    break

        # Pretty-print procedure name (capitalize first letter)
        proc_label = proc.capitalize()

        # Procedure header (midrule + multicolumn)
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{6}}{{c}}{{\textit{{{proc_label}}} ($n={n}$)}} \\")
        lines.append(r"\midrule")

        # ------------------------------------------------------------------
        # Collect raw values for each method x metric, so we can bold best.
        # Methods: direct, tps, morph, sd_img2img, landmarkdiff
        # ------------------------------------------------------------------
        method_names = [
            "Direct copy",
            "TPS-only",
            "Morphing",
            "SD1.5 Img2Img",
            r"\textbf{\method{}}",
        ]

        # For each metric, collect (formatted_str, raw_value) per method
        # Then bold the best and assemble rows.

        # Build a matrix: methods x metrics
        # raw_matrix[method_idx][metric_idx] = raw float or None
        # fmt_matrix[method_idx][metric_idx] = formatted string
        n_methods = len(method_names)
        n_metrics = len(TABLE1_METRICS)
        raw_matrix = [[None] * n_metrics for _ in range(n_methods)]
        fmt_matrix = [["--"] * n_metrics for _ in range(n_methods)]

        for mi, metric in enumerate(TABLE1_METRICS):
            dec = decimals_for_metric(metric)

            # ----- Direct copy -----
            # Direct copy has no FID, NME, or identity_sim in baseline data
            if metric in ("ssim", "lpips"):
                raw_matrix[0][mi] = get_baseline_val(baseline_data, proc, "direct", metric)
                fmt_matrix[0][mi] = fmt_val(raw_matrix[0][mi], dec)
            # FID, NME, identity_sim for direct copy -> "--"

            # ----- TPS-only -----
            if metric in ("ssim", "lpips", "nme"):
                raw_matrix[1][mi] = get_baseline_val(baseline_data, proc, "tps", metric)
                fmt_matrix[1][mi] = fmt_val(raw_matrix[1][mi], dec)

            # ----- Morphing -----
            if metric in ("ssim", "lpips"):
                raw_matrix[2][mi] = get_baseline_val(baseline_data, proc, "morph", metric)
                fmt_matrix[2][mi] = fmt_val(raw_matrix[2][mi], dec)

            # ----- SD1.5 Img2Img -----
            raw_matrix[3][mi] = get_sd_val(sd_data, proc, metric)
            fmt_matrix[3][mi] = fmt_val(raw_matrix[3][mi], dec)

            # ----- LandmarkDiff -----
            if ld_agg is not None and proc in ld_agg and metric in ld_agg[proc]:
                entry = ld_agg[proc][metric]
                raw_matrix[4][mi] = entry.get("mean")
                if has_multi_seed and entry.get("std", 0) > 0:
                    fmt_matrix[4][mi] = fmt_pm(entry.get("mean"), entry.get("std"), dec)
                else:
                    fmt_matrix[4][mi] = fmt_val(entry.get("mean"), dec)

        # Bold the best value per metric (column)
        for mi, metric in enumerate(TABLE1_METRICS):
            raw_col = [raw_matrix[mj][mi] for mj in range(n_methods)]
            fmt_col = [fmt_matrix[mj][mi] for mj in range(n_methods)]
            higher = HIGHER_IS_BETTER[metric]
            bolded = bold_best(fmt_col, higher, raw_col)
            for mj in range(n_methods):
                fmt_matrix[mj][mi] = bolded[mj]

        # Assemble rows
        for mj, name in enumerate(method_names):
            cells = " & ".join(fmt_matrix[mj])
            lines.append(f"{name} & {cells} \\\\")

    return "\n".join(lines)


def build_fitzpatrick_body(
    sd_data: dict | None,
    ld_agg: dict | None,
    baseline_data: dict | None,
) -> str:
    """Build the body of Table 3 (tab:fitzpatrick).

    Groups Fitzpatrick types I-II and V-VI together.  Shows two sections:
    TPS-only baseline (from sd_img2img or baseline_results fitzpatrick data)
    and LandmarkDiff.

    Table columns: LPIPS, SSIM, NME, ArcFace, n
    """
    lines = []

    # Fitzpatrick metrics in column order for this table
    fitz_metrics = ["lpips", "ssim", "nme", "identity_sim"]
    # Direction (for bolding within each section)
    group_order = ["I--II", "III", "IV", "V--VI"]

    # ------------------------------------------------------------------
    # Source 1: TPS-only / SD baseline Fitzpatrick data
    # The sd_img2img_baseline files contain a "fitzpatrick" section.
    # The baseline_results.json does NOT have fitzpatrick data.
    # So we use SD baseline data for the "TPS-only baseline" section,
    # since that is the only source with per-Fitzpatrick stratification.
    # ------------------------------------------------------------------
    sd_fitz = None
    if sd_data is not None:
        sd_fitz = sd_data.get("fitzpatrick")

    sd_groups = group_fitzpatrick(sd_fitz)

    # ------------------------------------------------------------------
    # Source 2: LandmarkDiff Fitzpatrick data (from aggregated results)
    # ------------------------------------------------------------------
    ld_fitz = None
    if ld_agg is not None:
        ld_fitz = ld_agg.get("fitzpatrick")

    ld_groups = group_fitzpatrick(ld_fitz)

    # ----- TPS-only baseline section -----
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{c}{\textit{TPS-only baseline}} \\")
    lines.append(r"\midrule")

    for group in group_order:
        g_data = sd_groups.get(group, {})
        cells = []
        for m in fitz_metrics:
            val = g_data.get(m, {}).get("mean")
            cells.append(fmt_val(val, 3))
        n = g_data.get(fitz_metrics[0], {}).get("n", 0)
        n_str = str(n) if n > 0 else "--"
        cells.append(n_str)
        line = f"Type {group}  & " + " & ".join(cells) + r" \\"
        lines.append(line)

    # ----- LandmarkDiff section -----
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{c}{\textit{\method{}}} \\")
    lines.append(r"\midrule")

    for group in group_order:
        g_data = ld_groups.get(group, {})
        cells = []
        for m in fitz_metrics:
            val = g_data.get(m, {}).get("mean")
            cells.append(fmt_val(val, 3))
        n = g_data.get(fitz_metrics[0], {}).get("n", 0)
        n_str = str(n) if n > 0 else "--"
        cells.append(n_str)
        line = f"Type {group}  & " + " & ".join(cells) + r" \\"
        lines.append(line)

    return "\n".join(lines)


# ====================================================================
# LaTeX table replacement engine
# ====================================================================


def replace_table_body(
    tex: str,
    label: str,
    new_body: str,
    table_start_marker: str = r"\toprule",
    table_end_marker: str = r"\bottomrule",
) -> tuple[str, bool]:
    """Replace the body of a LaTeX table identified by its \\label{}.

    Strategy:
      1. Find the \\label{<label>} in the document.
      2. From that position, search backward for the nearest \\toprule.
      3. From that \\toprule, search forward for the nearest \\bottomrule.
      4. Replace everything between \\toprule (exclusive) and \\bottomrule
         (exclusive) with new_body.

    This preserves the table header row (between \\toprule and first \\midrule
    in the new body) -- but since our new_body starts with \\midrule, the
    original header row is kept intact.

    Returns (modified_tex, success_bool).
    """
    # Find the label
    label_pattern = re.escape(r"\label{" + label + "}")
    label_match = re.search(label_pattern, tex)
    if label_match is None:
        print(f"  [WARN] Could not find \\label{{{label}}} in tex")
        return tex, False

    label_pos = label_match.start()

    # Find the \\toprule before the header row.
    # The header row is between \\toprule and the first \\midrule in the
    # original table.  We want to keep the header and replace everything
    # from the first \\midrule to \\bottomrule.

    # Find \\toprule after the label
    toprule_pos = tex.find(table_start_marker, label_pos)
    if toprule_pos == -1:
        print(f"  [WARN] Could not find {table_start_marker} after \\label{{{label}}}")
        return tex, False

    # Find the header row: everything from \\toprule to the first \\midrule
    # We need to find the first \\midrule after \\toprule
    first_midrule = tex.find(r"\midrule", toprule_pos + len(table_start_marker))
    if first_midrule == -1:
        print(f"  [WARN] Could not find first \\midrule after {table_start_marker}")
        return tex, False

    # Find \\bottomrule
    bottomrule_pos = tex.find(table_end_marker, first_midrule)
    if bottomrule_pos == -1:
        print(f"  [WARN] Could not find {table_end_marker} after first \\midrule")
        return tex, False

    # The header row ends at the first \midrule (which is part of the header).
    # Our new_body starts with \midrule, so we replace from the first \midrule
    # to just before \bottomrule.
    # Keep: tex[:first_midrule] + new_body + "\n" + tex[bottomrule_pos:]
    modified = tex[:first_midrule] + new_body + "\n" + tex[bottomrule_pos:]

    return modified, True


# ====================================================================
# Main logic
# ====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Auto-update LaTeX tables in paper/main.tex with evaluation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/update_paper_tables.py
    python scripts/update_paper_tables.py --dry-run
    python scripts/update_paper_tables.py --paper paper/alt_main.tex
        """,
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=ROOT / "paper" / "main.tex",
        help="Path to the LaTeX file to update (default: paper/main.tex)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "paper",
        help="Directory containing result JSON files (default: paper/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without writing the file",
    )
    args = parser.parse_args()

    paper_path: Path = args.paper
    results_dir: Path = args.results_dir

    # ------------------------------------------------------------------
    # Step 1: Load all available result files
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading result files...")
    print("=" * 60)

    # 1a. Baseline results (TPS, morphing, direct copy)
    baseline_data = load_json(results_dir / "baseline_results.json")

    # 1b. SD1.5 img2img baselines at various strengths
    #     We look for s0_3, s0_5, s0_7 (strengths 0.3, 0.5, 0.7)
    sd_strengths = [0.3, 0.5, 0.7]
    sd_files: dict[float, dict | None] = {}
    for s in sd_strengths:
        # Filename convention: sd_img2img_baseline_s0_3.json for strength=0.3
        fname = f"sd_img2img_baseline_s{str(s).replace('.', '_')}.json"
        sd_files[s] = load_json(results_dir / fname)

    # Pick the best SD strength (highest overall SSIM)
    best_sd_strength, best_sd_data = select_best_sd_strength(sd_files)

    # 1c. LandmarkDiff per-seed evaluation results
    seeds = [42, 123, 456]
    seed_files: dict[int, dict | None] = {}
    for seed in seeds:
        seed_files[seed] = load_json(results_dir / f"eval_results_seed{seed}.json")

    # 1d. Aggregated LandmarkDiff results (preferred if available)
    ld_aggregated = load_json(results_dir / "eval_results_aggregated.json")

    # Determine whether we have multi-seed data
    n_valid_seeds = sum(1 for d in seed_files.values() if d is not None)
    has_multi_seed = n_valid_seeds > 1

    # If aggregated file exists, use it; otherwise compute from seeds
    if ld_aggregated is not None:
        print("  [INFO] Using pre-computed aggregated results")
        ld_agg = ld_aggregated
    elif n_valid_seeds > 0:
        print(f"  [INFO] No aggregated file; computing from {n_valid_seeds} seed(s)")
        try:
            ld_agg = aggregate_seeds(seed_files)
        except ImportError:
            print("  [WARN] numpy not available; cannot aggregate seeds")
            # Fall back to the first available seed
            for s, d in seed_files.items():
                if d is not None:
                    ld_agg = d
                    has_multi_seed = False
                    break
            else:
                ld_agg = None
    else:
        print("  [INFO] No LandmarkDiff evaluation results available")
        ld_agg = None

    # ------------------------------------------------------------------
    # Step 2: Read the LaTeX file
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Reading {paper_path.relative_to(ROOT)}...")
    print("=" * 60)

    if not paper_path.exists():
        print(f"  [ERROR] {paper_path} does not exist!")
        sys.exit(1)

    tex_original = paper_path.read_text(encoding="utf-8")
    tex = tex_original  # working copy

    # ------------------------------------------------------------------
    # Step 3: Build and replace Table 1 (tab:results)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Building Table 1 (tab:results)...")
    print("=" * 60)

    table1_body = build_table1_body(
        baseline_data=baseline_data,
        sd_data=best_sd_data,
        ld_agg=ld_agg,
        has_multi_seed=has_multi_seed,
    )
    tex, t1_ok = replace_table_body(tex, "tab:results", table1_body)
    if t1_ok:
        print("  [OK]   Table 1 (tab:results) updated successfully")
    else:
        print("  [FAIL] Table 1 (tab:results) could not be updated")

    # ------------------------------------------------------------------
    # Step 4: Build and replace Table 3 (tab:fitzpatrick)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Building Table 3 (tab:fitzpatrick)...")
    print("=" * 60)

    fitz_body = build_fitzpatrick_body(
        sd_data=best_sd_data,
        ld_agg=ld_agg,
        baseline_data=baseline_data,
    )
    tex, t3_ok = replace_table_body(tex, "tab:fitzpatrick", fitz_body)
    if t3_ok:
        print("  [OK]   Table 3 (tab:fitzpatrick) updated successfully")
    else:
        print("  [FAIL] Table 3 (tab:fitzpatrick) could not be updated")

    # ------------------------------------------------------------------
    # Step 5: Write back or show diff
    # ------------------------------------------------------------------
    print()
    print("=" * 60)

    if tex == tex_original:
        print("No changes detected. LaTeX file is already up to date.")
        return

    if args.dry_run:
        print("[DRY RUN] Would write changes to", paper_path)
        # Show a simple diff summary
        orig_lines = tex_original.splitlines()
        new_lines = tex.splitlines()
        print(f"  Original: {len(orig_lines)} lines")
        print(f"  Modified: {len(new_lines)} lines")
        print(
            f"  Changed:  {sum(1 for a, b in zip(orig_lines, new_lines, strict=False) if a != b)} lines differ"
        )
        print()
        # Print the first few changed lines for inspection
        n_shown = 0
        for i, (a, b) in enumerate(zip(orig_lines, new_lines, strict=False)):
            if a != b and n_shown < 20:
                print(f"  Line {i + 1}:")
                print(f"    - {a}")
                print(f"    + {b}")
                n_shown += 1
    else:
        paper_path.write_text(tex, encoding="utf-8")
        print(f"Written updated tables to {paper_path}")

    # ------------------------------------------------------------------
    # Step 6: Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"  Table 1 (tab:results):      {'UPDATED' if t1_ok else 'SKIPPED'}")
    print(f"  Table 3 (tab:fitzpatrick):  {'UPDATED' if t3_ok else 'SKIPPED'}")
    print()

    # Report data availability
    print("  Data sources:")
    print(f"    baseline_results.json:       {'loaded' if baseline_data else 'MISSING'}")
    for s in sd_strengths:
        fname = f"sd_img2img_baseline_s{str(s).replace('.', '_')}.json"
        loaded = sd_files[s] is not None
        best = " <-- BEST" if s == best_sd_strength and loaded else ""
        print(f"    {fname}: {'loaded' if loaded else 'MISSING'}{best}")
    for seed in seeds:
        loaded = seed_files[seed] is not None
        print(f"    eval_results_seed{seed}.json:   {'loaded' if loaded else 'MISSING'}")
    print(f"    eval_results_aggregated.json: {'loaded' if ld_aggregated else 'MISSING'}")
    print()

    # Report per-procedure LandmarkDiff values (for quick sanity check)
    if ld_agg is not None:
        print("  LandmarkDiff results used:")
        for proc in PROCEDURES:
            proc_data = ld_agg.get(proc, {})
            ssim = proc_data.get("ssim", {}).get("mean")
            lpips = proc_data.get("lpips", {}).get("mean")
            nme = proc_data.get("nme", {}).get("mean")
            idsim = proc_data.get("identity_sim", {}).get("mean")
            print(
                f"    {proc:15s}: SSIM={fmt_val(ssim, 3):>6s}  "
                f"LPIPS={fmt_val(lpips, 3):>6s}  "
                f"NME={fmt_val(nme, 3):>6s}  "
                f"ArcFace={fmt_val(idsim, 3):>6s}"
            )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
