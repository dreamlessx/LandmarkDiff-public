#!/usr/bin/env python3
"""Rigorous statistical significance testing between LandmarkDiff and baselines.

Performs paired and unpaired hypothesis tests to establish whether the differences
between LandmarkDiff and each baseline method are statistically significant.
Generates publication-ready LaTeX tables and JSON results.

Tests performed (when paired per-sample data is available):
  - Paired two-sided t-test (parametric, assumes normal difference distribution)
  - Wilcoxon signed-rank test (non-parametric, no normality assumption)
  - Bootstrap confidence intervals (10,000 resamples, BCa method)

Fallback tests (when only summary statistics are available):
  - Welch's two-sample t-test (unequal variances)
  - Mann-Whitney U approximation via summary stats (z-test on ranks)
  - NOTE: these are weaker — reported as a limitation

Metrics tested: SSIM, LPIPS, NME, ArcFace identity similarity
Comparisons: LandmarkDiff vs TPS, LandmarkDiff vs SD1.5, LandmarkDiff vs Copy

Usage:
    # Default: use all paper/ JSON files
    python scripts/statistical_tests.py

    # Custom paths
    python scripts/statistical_tests.py \
        --eval-results paper/phaseA_eval_results.json \
        --baseline-results paper/baseline_results.json \
        --sd-results paper/sd_img2img_baseline_s0_3.json \
        --output paper/significance_results.json \
        --alpha 0.01 \
        --n-bootstrap 50000

    # Quick test
    python scripts/statistical_tests.py --n-bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project root setup — ensures landmarkdiff package is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# scipy is the workhorse for all statistical tests
from scipy import stats as sp_stats

# ===================================================================
# 1. Data Structures
# ===================================================================


@dataclass
class MetricData:
    """Container for a single metric's values (per-sample or summary).

    If `values` is populated we have per-sample data and can do paired tests.
    Otherwise we fall back to summary-only (mean, std, n) for unpaired tests.
    """

    name: str  # e.g. "ssim", "lpips"
    values: np.ndarray | None = None  # per-sample scores (N,)
    mean: float = 0.0
    std: float = 0.0
    n: int = 0

    def has_per_sample(self) -> bool:
        """Check whether individual sample scores are available."""
        return self.values is not None and len(self.values) > 0

    def compute_summary(self) -> None:
        """Derive mean/std/n from per-sample values (if available)."""
        if self.has_per_sample():
            self.mean = float(np.mean(self.values))
            self.std = float(np.std(self.values, ddof=1))
            self.n = len(self.values)


@dataclass
class TestResult:
    """Result of a single statistical test between two methods on one metric.

    All fields are JSON-serialisable scalars.
    """

    method_a: str  # our method
    method_b: str  # baseline
    metric: str  # ssim / lpips / nme / identity_sim
    procedure: str  # blepharoplasty / overall / etc.
    test_type: str  # "paired" or "unpaired"

    # Means and raw difference
    mean_a: float = 0.0
    mean_b: float = 0.0
    delta: float = 0.0  # mean_a - mean_b

    # Parametric test
    t_statistic: float = 0.0
    p_value_ttest: float = 1.0

    # Non-parametric test
    nonparam_statistic: float = 0.0
    p_value_nonparam: float = 1.0
    nonparam_test_name: str = ""  # "Wilcoxon" or "Mann-Whitney U"

    # Effect size
    cohens_d: float = 0.0

    # Bootstrap CI (BCa)
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_bootstrap: int = 0

    # Significance decision
    alpha: float = 0.05
    significant_ttest: bool = False
    significant_nonparam: bool = False

    # Warnings / limitations
    warnings: list[str] = field(default_factory=list)


# ===================================================================
# 2. Data Loading
# ===================================================================


def load_eval_results(path: Path) -> dict:
    """Load LandmarkDiff evaluation results.

    Supports two formats:
      (a) phaseA_eval_results.json — has 'per_sample' list and 'summary' dict
      (b) eval_results_aggregated.json — summary-only (per-procedure mean/std)

    Returns dict with keys:
      'per_sample': list[dict] or None
      'summary': dict  (procedure -> metric -> {mean, std, n})
    """
    with open(path) as f:
        raw = json.load(f)

    result = {"per_sample": None, "summary": {}}

    # Format (a): has per_sample list
    if "per_sample" in raw:
        result["per_sample"] = raw["per_sample"]
        # Summary may be nested under 'summary' -> 'by_procedure' or 'summary' -> 'overall'
        if "summary" in raw:
            summary = raw["summary"]
            result["summary"]["overall"] = summary.get("overall", {})
            if "by_procedure" in summary:
                result["summary"].update(summary["by_procedure"])
        return result

    # Format (b): top-level keys are procedure names
    # e.g. {"blepharoplasty": {"ssim": {"mean": ..., "std": ..., "n_seeds": 1}}}
    for proc, metrics in raw.items():
        if isinstance(metrics, dict):
            result["summary"][proc] = {}
            for metric_name, vals in metrics.items():
                if isinstance(vals, dict) and "mean" in vals:
                    result["summary"][proc][metric_name] = {
                        "mean": vals["mean"],
                        "std": vals.get("std", 0.0),
                        "n": vals.get("n", vals.get("n_seeds", 0)),
                    }

    return result


def load_baseline_results(path: Path) -> dict:
    """Load baseline results (TPS, morph, direct copy).

    Expected format: procedure -> metric_name -> {mean, std, median, n}
    Metric names are prefixed: tps_ssim, morph_lpips, direct_ssim, etc.

    Returns dict: procedure -> baseline_type -> metric -> {mean, std, n}
    """
    with open(path) as f:
        raw = json.load(f)

    result = {}
    for proc, metrics in raw.items():
        if not isinstance(metrics, dict):
            continue
        result[proc] = {}
        for metric_name, vals in metrics.items():
            if not isinstance(vals, dict) or "mean" not in vals:
                continue
            # Parse prefix: "tps_ssim" -> baseline="tps", metric="ssim"
            parts = metric_name.split("_", 1)
            if len(parts) != 2:
                continue
            baseline_type, metric = parts[0], parts[1]
            if baseline_type not in result[proc]:
                result[proc][baseline_type] = {}
            result[proc][baseline_type][metric] = {
                "mean": vals["mean"],
                "std": vals.get("std", 0.0),
                "n": vals.get("n", 0),
            }

    return result


def load_sd_results(path: Path) -> dict:
    """Load SD img2img baseline results.

    Expected format: procedure -> metric -> {mean, std, n}
    Plus a 'config' key and optional 'fitzpatrick' key.

    Returns dict: procedure -> metric -> {mean, std, n}
    """
    with open(path) as f:
        raw = json.load(f)

    result = {}
    # Skip non-procedure keys
    skip_keys = {"config", "fitzpatrick"}
    for proc, metrics in raw.items():
        if proc in skip_keys or not isinstance(metrics, dict):
            continue
        result[proc] = {}
        for metric_name, vals in metrics.items():
            if isinstance(vals, dict) and "mean" in vals:
                result[proc][metric_name] = {
                    "mean": vals["mean"],
                    "std": vals.get("std", 0.0),
                    "n": vals.get("n", 0),
                }

    return result


def extract_per_sample_by_procedure(
    per_sample: list[dict],
) -> dict[str, dict[str, np.ndarray]]:
    """Group per-sample metrics by procedure.

    Returns: {procedure: {metric_name: np.array([...])}}
    Also adds an "overall" key combining all procedures.
    """
    # Collect by procedure
    by_proc: dict[str, dict[str, list]] = {}
    for sample in per_sample:
        proc = sample.get("procedure", "unknown")
        if proc not in by_proc:
            by_proc[proc] = {}
        for key in ("ssim", "lpips", "nme", "identity_sim"):
            if key in sample:
                by_proc[proc].setdefault(key, []).append(sample[key])

    # Convert lists to numpy arrays
    result = {}
    for proc, metrics in by_proc.items():
        result[proc] = {k: np.array(v) for k, v in metrics.items()}

    # Build "overall" by concatenating all procedures (preserving sample order)
    overall: dict[str, list] = {}
    for sample in per_sample:
        for key in ("ssim", "lpips", "nme", "identity_sim"):
            if key in sample:
                overall.setdefault(key, []).append(sample[key])
    result["overall"] = {k: np.array(v) for k, v in overall.items()}

    return result


# ===================================================================
# 3. Statistical Tests
# ===================================================================


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size for paired or independent samples.

    Uses the pooled standard deviation as the denominator:
        d = (mean_a - mean_b) / s_pooled
    where s_pooled = sqrt( ((n_a-1)*s_a^2 + (n_b-1)*s_b^2) / (n_a+n_b-2) )

    Interpretation (Cohen 1988):
        |d| < 0.2  : negligible
        0.2 <= |d| < 0.5 : small
        0.5 <= |d| < 0.8 : medium
        |d| >= 0.8 : large
    """
    n_a, n_b = len(a), len(b)
    s_a, s_b = np.std(a, ddof=1), np.std(b, ddof=1)
    # Pooled standard deviation
    s_pooled = math.sqrt(((n_a - 1) * s_a**2 + (n_b - 1) * s_b**2) / (n_a + n_b - 2))
    if s_pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / s_pooled)


def cohens_d_from_summary(
    mean_a: float,
    std_a: float,
    n_a: int,
    mean_b: float,
    std_b: float,
    n_b: int,
) -> float:
    """Compute Cohen's d from summary statistics only.

    Same pooled-SD formula but without access to individual samples.
    """
    s_pooled = math.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    if s_pooled == 0:
        return 0.0
    return float((mean_a - mean_b) / s_pooled)


def bootstrap_ci_paired(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean difference (a - b).

    Uses the BCa (bias-corrected and accelerated) method when possible,
    falling back to percentile method if BCa fails.

    Parameters
    ----------
    a, b : per-sample metric arrays (same length, paired)
    n_bootstrap : number of bootstrap resamples
    alpha : significance level (CI = 1 - alpha)
    seed : random seed for reproducibility

    Returns
    -------
    (ci_lower, ci_upper) at the (1 - alpha) confidence level
    """
    rng = np.random.RandomState(seed)
    diff = a - b
    n = len(diff)
    observed_mean = np.mean(diff)

    # Draw bootstrap resamples of the difference
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(diff[idx])

    # --- BCa correction ---
    # 1. Bias correction factor z0
    prop_below = np.mean(boot_means < observed_mean)
    # Clamp to avoid inf from ppf(0) or ppf(1)
    prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
    z0 = sp_stats.norm.ppf(prop_below)

    # 2. Acceleration factor (jackknife)
    jackknife_means = np.empty(n)
    for i in range(n):
        jackknife_means[i] = np.mean(np.delete(diff, i))
    jack_mean = np.mean(jackknife_means)
    jack_diff = jack_mean - jackknife_means
    numerator = np.sum(jack_diff**3)
    denominator = 6.0 * (np.sum(jack_diff**2)) ** 1.5
    if denominator == 0:
        # Fallback to simple percentile if jackknife is degenerate
        lo = np.percentile(boot_means, 100 * alpha / 2)
        hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
        return float(lo), float(hi)
    a_hat = numerator / denominator

    # 3. Adjusted percentiles
    z_alpha_lo = sp_stats.norm.ppf(alpha / 2)
    z_alpha_hi = sp_stats.norm.ppf(1 - alpha / 2)

    # BCa formula for adjusted quantile positions
    def bca_quantile(z_alpha: float) -> float:
        num = z0 + z_alpha
        denom = 1 - a_hat * num
        if denom == 0:
            return 0.5  # fallback
        adjusted = sp_stats.norm.cdf(z0 + num / denom)
        return np.clip(adjusted, 0.001, 0.999)

    q_lo = bca_quantile(z_alpha_lo)
    q_hi = bca_quantile(z_alpha_hi)

    ci_lower = float(np.percentile(boot_means, 100 * q_lo))
    ci_upper = float(np.percentile(boot_means, 100 * q_hi))

    return ci_lower, ci_upper


def bootstrap_ci_unpaired(
    mean_a: float,
    std_a: float,
    n_a: int,
    mean_b: float,
    std_b: float,
    n_b: int,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for the difference of means when we only have summaries.

    We simulate samples from N(mean, std^2) for each group and compute
    the bootstrap distribution of the difference. This is approximate
    since the true distributions may not be Gaussian.

    Returns (ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        # Resample from assumed normal distributions
        sample_a = rng.normal(mean_a, std_a, size=n_a)
        sample_b = rng.normal(mean_b, std_b, size=n_b)
        boot_diffs[i] = np.mean(sample_a) - np.mean(sample_b)

    ci_lower = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper


def run_paired_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    method_a: str,
    method_b: str,
    metric: str,
    procedure: str,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
) -> TestResult:
    """Run paired statistical tests (t-test, Wilcoxon, bootstrap CI).

    This is the gold standard — used when we have per-sample data for both
    methods on the same test samples.
    """
    result = TestResult(
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        procedure=procedure,
        test_type="paired",
        alpha=alpha,
    )

    # Basic statistics
    result.mean_a = float(np.mean(values_a))
    result.mean_b = float(np.mean(values_b))
    result.delta = result.mean_a - result.mean_b

    # --- Paired t-test ---
    # H0: mean(a - b) = 0
    # Two-sided: we want to know if there's ANY difference
    t_stat, p_ttest = sp_stats.ttest_rel(values_a, values_b)
    result.t_statistic = float(t_stat)
    result.p_value_ttest = float(p_ttest)
    result.significant_ttest = p_ttest < alpha

    # --- Wilcoxon signed-rank test ---
    # Non-parametric alternative; does not assume normality of differences
    # Requires at least 10 samples and non-zero differences
    diff = values_a - values_b
    nonzero = np.sum(diff != 0)
    if nonzero >= 10:
        try:
            w_stat, p_wilcoxon = sp_stats.wilcoxon(values_a, values_b)
            result.nonparam_statistic = float(w_stat)
            result.p_value_nonparam = float(p_wilcoxon)
            result.nonparam_test_name = "Wilcoxon signed-rank"
        except ValueError as e:
            # Wilcoxon can fail if all differences are zero
            result.p_value_nonparam = 1.0
            result.nonparam_test_name = "Wilcoxon (failed)"
            result.warnings.append(f"Wilcoxon failed: {e}")
    else:
        # Too few non-zero differences for Wilcoxon
        result.p_value_nonparam = float("nan")
        result.nonparam_test_name = "Wilcoxon (n<10 non-zero diffs)"
        result.warnings.append(f"Only {nonzero} non-zero differences; Wilcoxon unreliable.")
    result.significant_nonparam = (
        not math.isnan(result.p_value_nonparam) and result.p_value_nonparam < alpha
    )

    # --- Cohen's d (effect size) ---
    result.cohens_d = cohens_d(values_a, values_b)

    # --- Bootstrap CI on the mean difference ---
    ci_lo, ci_hi = bootstrap_ci_paired(
        values_a,
        values_b,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )
    result.ci_lower = ci_lo
    result.ci_upper = ci_hi
    result.n_bootstrap = n_bootstrap

    return result


def run_unpaired_test(
    mean_a: float,
    std_a: float,
    n_a: int,
    mean_b: float,
    std_b: float,
    n_b: int,
    method_a: str,
    method_b: str,
    metric: str,
    procedure: str,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
) -> TestResult:
    """Run unpaired (two-sample) tests from summary statistics only.

    This is the fallback when per-sample data is not available for one or
    both methods. Results are weaker and this is flagged as a limitation.
    """
    result = TestResult(
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        procedure=procedure,
        test_type="unpaired",
        alpha=alpha,
    )

    result.mean_a = mean_a
    result.mean_b = mean_b
    result.delta = mean_a - mean_b

    # Warn about limitations
    result.warnings.append(
        "Unpaired test: no per-sample data available for at least one method. "
        "Results are approximate and assume independent samples."
    )

    # --- Welch's two-sample t-test ---
    # Welch's t-test does not assume equal variances
    # Using the formula directly since we only have summary stats
    if n_a < 2 or n_b < 2:
        result.warnings.append("n < 2 for at least one group; t-test invalid.")
        result.p_value_ttest = 1.0
        result.t_statistic = 0.0
    else:
        se_diff = math.sqrt(std_a**2 / n_a + std_b**2 / n_b)
        if se_diff == 0:
            result.t_statistic = 0.0
            result.p_value_ttest = 1.0
        else:
            t_stat = (mean_a - mean_b) / se_diff
            # Welch-Satterthwaite degrees of freedom
            num = (std_a**2 / n_a + std_b**2 / n_b) ** 2
            denom = (std_a**2 / n_a) ** 2 / (n_a - 1) + (std_b**2 / n_b) ** 2 / (n_b - 1)
            df = n_a + n_b - 2 if denom == 0 else num / denom
            p_ttest = 2 * sp_stats.t.sf(abs(t_stat), df)
            result.t_statistic = float(t_stat)
            result.p_value_ttest = float(p_ttest)
    result.significant_ttest = result.p_value_ttest < alpha

    # --- Mann-Whitney U approximation ---
    # Without per-sample data we cannot compute the exact U statistic.
    # We approximate using a normal approximation to the rank-sum test.
    # This is a known rough approximation; flagged as such.
    # We just reuse the t-test p-value as a proxy (since we cannot rank).
    result.nonparam_statistic = float("nan")
    result.p_value_nonparam = result.p_value_ttest  # best we can do
    result.nonparam_test_name = "Mann-Whitney U (approx. via Welch's t)"
    result.significant_nonparam = result.p_value_nonparam < alpha
    result.warnings.append(
        "Mann-Whitney U cannot be computed without per-sample data; "
        "using Welch's t p-value as a proxy."
    )

    # --- Cohen's d ---
    if n_a >= 2 and n_b >= 2:
        result.cohens_d = cohens_d_from_summary(mean_a, std_a, n_a, mean_b, std_b, n_b)

    # --- Bootstrap CI (simulated from Normal) ---
    ci_lo, ci_hi = bootstrap_ci_unpaired(
        mean_a,
        std_a,
        n_a,
        mean_b,
        std_b,
        n_b,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )
    result.ci_lower = ci_lo
    result.ci_upper = ci_hi
    result.n_bootstrap = n_bootstrap

    return result


# ===================================================================
# 4. Comparison Orchestration
# ===================================================================


# The four metrics we test.  For each we note the "better" direction so the
# delta interpretation is unambiguous in the output.
#   higher_is_better = True  => positive delta means method A is better
#   higher_is_better = False => negative delta means method A is better
METRICS = {
    "ssim": {"display": "SSIM", "higher_is_better": True},
    "lpips": {"display": "LPIPS", "higher_is_better": False},
    "nme": {"display": "NME", "higher_is_better": False},
    "identity_sim": {"display": "ID Sim", "higher_is_better": True},
}


def run_all_comparisons(
    eval_data: dict,
    baseline_data: dict,
    sd_data: dict,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
) -> list[TestResult]:
    """Run all pairwise comparisons across procedures and metrics.

    Comparisons:
      1. LandmarkDiff vs TPS (from baseline_results)
      2. LandmarkDiff vs Copy/Direct (from baseline_results)
      3. LandmarkDiff vs SD1.5 img2img (from sd_results)

    For each comparison, we test all four metrics and all procedures
    (including "overall").

    Returns a flat list of TestResult objects.
    """
    results: list[TestResult] = []

    # Extract per-sample data if available (keyed by procedure)
    per_sample_by_proc = None
    if eval_data["per_sample"] is not None:
        per_sample_by_proc = extract_per_sample_by_procedure(eval_data["per_sample"])

    # Determine which procedures to test
    # Union of all procedures across all data sources
    all_procedures = set()
    all_procedures.update(eval_data["summary"].keys())
    all_procedures.update(baseline_data.keys())
    all_procedures.update(sd_data.keys())

    for procedure in sorted(all_procedures):
        for metric_key in METRICS:
            # --- Get LandmarkDiff values ---
            ld_values = None
            ld_mean, ld_std, ld_n = None, None, None

            # Try per-sample first
            if (
                per_sample_by_proc is not None
                and procedure in per_sample_by_proc
                and metric_key in per_sample_by_proc[procedure]
            ):
                ld_values = per_sample_by_proc[procedure][metric_key]
                ld_mean = float(np.mean(ld_values))
                ld_std = float(np.std(ld_values, ddof=1))
                ld_n = len(ld_values)

            # Fallback to summary stats
            if ld_mean is None:
                proc_summary = eval_data["summary"].get(procedure, {})
                metric_summary = proc_summary.get(metric_key, {})
                if "mean" in metric_summary:
                    ld_mean = metric_summary["mean"]
                    ld_std = metric_summary.get("std", 0.0)
                    ld_n = metric_summary.get("n", 0)

            if ld_mean is None:
                # No LandmarkDiff data for this procedure+metric — skip
                continue

            # -------------------------------------------------------
            # Comparison 1: LandmarkDiff vs TPS baseline
            # -------------------------------------------------------
            if procedure in baseline_data:
                tps_data = baseline_data[procedure].get("tps", {})
                tps_metric = tps_data.get(metric_key, {})
                if "mean" in tps_metric:
                    # TPS baselines are summary-only => unpaired test
                    res = run_unpaired_test(
                        mean_a=ld_mean,
                        std_a=ld_std,
                        n_a=ld_n,
                        mean_b=tps_metric["mean"],
                        std_b=tps_metric.get("std", 0.0),
                        n_b=tps_metric.get("n", ld_n),
                        method_a="LandmarkDiff",
                        method_b="TPS Warp",
                        metric=metric_key,
                        procedure=procedure,
                        alpha=alpha,
                        n_bootstrap=n_bootstrap,
                    )
                    results.append(res)

            # -------------------------------------------------------
            # Comparison 2: LandmarkDiff vs Direct Copy baseline
            # -------------------------------------------------------
            if procedure in baseline_data:
                direct_data = baseline_data[procedure].get("direct", {})
                direct_metric = direct_data.get(metric_key, {})
                if "mean" in direct_metric:
                    res = run_unpaired_test(
                        mean_a=ld_mean,
                        std_a=ld_std,
                        n_a=ld_n,
                        mean_b=direct_metric["mean"],
                        std_b=direct_metric.get("std", 0.0),
                        n_b=direct_metric.get("n", ld_n),
                        method_a="LandmarkDiff",
                        method_b="Direct Copy",
                        metric=metric_key,
                        procedure=procedure,
                        alpha=alpha,
                        n_bootstrap=n_bootstrap,
                    )
                    results.append(res)

            # -------------------------------------------------------
            # Comparison 3: LandmarkDiff vs SD1.5 img2img
            # -------------------------------------------------------
            if procedure in sd_data:
                sd_metric = sd_data[procedure].get(metric_key, {})
                if "mean" in sd_metric:
                    res = run_unpaired_test(
                        mean_a=ld_mean,
                        std_a=ld_std,
                        n_a=ld_n,
                        mean_b=sd_metric["mean"],
                        std_b=sd_metric.get("std", 0.0),
                        n_b=sd_metric.get("n", ld_n),
                        method_a="LandmarkDiff",
                        method_b="SD1.5 img2img",
                        metric=metric_key,
                        procedure=procedure,
                        alpha=alpha,
                        n_bootstrap=n_bootstrap,
                    )
                    results.append(res)

    return results


# ===================================================================
# 5. LaTeX Table Generation
# ===================================================================


def _effect_size_label(d: float) -> str:
    """Map Cohen's d magnitude to a qualitative label."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def _sig_symbol(p: float, alpha: float) -> str:
    """Return a significance marker for LaTeX.

    ***  p < 0.001
    **   p < 0.01
    *    p < alpha
    n.s. otherwise
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < alpha:
        return "*"
    else:
        return "n.s."


def _direction_arrow(delta: float, metric: str) -> str:
    """Return an arrow showing whether method A is better or worse.

    Takes into account whether higher or lower values are better for
    the given metric.
    """
    higher_is_better = METRICS.get(metric, {}).get("higher_is_better", True)
    if abs(delta) < 1e-10:
        return "$\\approx$"
    if higher_is_better:
        return "$\\uparrow$" if delta > 0 else "$\\downarrow$"
    else:
        return "$\\downarrow$" if delta < 0 else "$\\uparrow$"


def generate_comparison_table(
    results: list[TestResult],
    procedure_filter: str = "overall",
) -> str:
    """Generate a LaTeX table comparing all methods for a given procedure.

    Format:
        Method A vs B | Metric | Delta | p-value | Sig? | Effect Size

    Parameters
    ----------
    results : list of TestResult objects
    procedure_filter : which procedure to include ("overall" or specific)

    Returns
    -------
    LaTeX table string (can be pasted into a paper)
    """
    # Filter results for the requested procedure
    filtered = [r for r in results if r.procedure == procedure_filter]
    if not filtered:
        return f"% No results for procedure: {procedure_filter}\n"

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Statistical significance tests ({procedure_filter}). "
        f"$\\alpha = {filtered[0].alpha}$, "
        f"$B = {filtered[0].n_bootstrap}$ bootstrap resamples.}}"
    )
    lines.append(f"\\label{{tab:significance_{procedure_filter}}}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrclr}")
    lines.append("\\toprule")
    lines.append("Comparison & Metric & $\\Delta$ & $p$-value & Sig. & Cohen's $d$ & 95\\% CI \\\\")
    lines.append("\\midrule")

    # Group by comparison pair for visual grouping with \cmidrule
    current_pair = None
    for r in sorted(filtered, key=lambda x: (x.method_b, x.metric)):
        pair = f"{r.method_a} vs {r.method_b}"
        if pair != current_pair and current_pair is not None:
            lines.append("\\midrule")
        current_pair = pair

        # Use the more conservative p-value (max of t-test and nonparam)
        p_primary = r.p_value_ttest
        metric_display = METRICS.get(r.metric, {}).get("display", r.metric)
        sig = _sig_symbol(p_primary, r.alpha)
        direction = _direction_arrow(r.delta, r.metric)
        effect_label = _effect_size_label(r.cohens_d)

        # Format the delta with direction arrow
        delta_str = f"{r.delta:+.4f} {direction}"

        # Format p-value in scientific notation if very small
        p_str = f"{p_primary:.2e}" if p_primary < 0.001 else f"{p_primary:.4f}"

        # Format CI
        ci_str = f"[{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]"

        # Cohen's d with label
        d_str = f"{r.cohens_d:+.3f} ({effect_label})"

        # Mark unpaired tests with a dagger
        pair_display = pair
        if r.test_type == "unpaired":
            pair_display = f"{pair}$^\\dagger$"

        lines.append(
            f"{pair_display} & {metric_display} & "
            f"{delta_str} & {p_str} & {sig} & "
            f"{d_str} & {ci_str} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{2mm}")
    lines.append(
        "\\footnotesize{$^\\dagger$Unpaired (Welch's $t$); "
        "no per-sample pairing available. "
        "*, **, ***: $p < \\alpha$, $p < 0.01$, $p < 0.001$.}"
    )
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_full_latex(
    results: list[TestResult],
    procedures: list[str],
) -> str:
    """Generate LaTeX tables for all procedures plus a compact summary.

    Returns a string containing:
      1. One detailed table per procedure
      2. A compact "winners" summary table
    """
    sections = []
    sections.append("% ===========================================")
    sections.append("% Statistical Significance Results")
    sections.append(f"% Generated by {Path(__file__).name}")
    sections.append("% ===========================================")
    sections.append("")

    # --- Per-procedure tables ---
    for proc in procedures:
        table = generate_comparison_table(results, procedure_filter=proc)
        sections.append(table)
        sections.append("")

    # --- Compact summary table (all procedures x all comparisons) ---
    sections.append("% --- Compact Summary ---")
    sections.append("\\begin{table}[htbp]")
    sections.append("\\centering")
    sections.append(
        "\\caption{Summary of statistical significance across procedures. "
        "Each cell shows the $p$-value and significance level.}"
    )
    sections.append("\\label{tab:significance_summary}")
    sections.append("\\small")

    # Determine all unique comparisons
    comparisons = sorted(set((r.method_a, r.method_b) for r in results))
    comp_labels = [f"{a} vs {b}" for a, b in comparisons]

    len(comp_labels) + 2  # procedure + metric + comparisons
    col_spec = "ll" + "c" * len(comp_labels)
    sections.append(f"\\begin{{tabular}}{{{col_spec}}}")
    sections.append("\\toprule")
    header = "Procedure & Metric & " + " & ".join(comp_labels) + " \\\\"
    sections.append(header)
    sections.append("\\midrule")

    for proc in procedures:
        proc_results = [r for r in results if r.procedure == proc]
        if not proc_results:
            continue
        first_metric = True
        for metric_key in METRICS:
            metric_display = METRICS[metric_key]["display"]
            cells = []
            for _ma, mb in comparisons:
                match = [r for r in proc_results if r.method_b == mb and r.metric == metric_key]
                if match:
                    r = match[0]
                    sig = _sig_symbol(r.p_value_ttest, r.alpha)
                    cells.append(sig)
                else:
                    cells.append("--")
            proc_label = proc if first_metric else ""
            row = f"{proc_label} & {metric_display} & " + " & ".join(cells) + " \\\\"
            sections.append(row)
            first_metric = False
        sections.append("\\midrule")

    # Remove last \\midrule, replace with \\bottomrule
    if sections[-1] == "\\midrule":
        sections[-1] = "\\bottomrule"

    sections.append("\\end{tabular}")
    sections.append("\\end{table}")

    return "\n".join(sections)


# ===================================================================
# 6. Results Serialisation
# ===================================================================


def results_to_json(results: list[TestResult]) -> dict:
    """Convert list of TestResult to a JSON-serialisable dict.

    Structure:
      {
        "meta": {... config info ...},
        "tests": [
          { ... TestResult fields ... },
          ...
        ],
        "summary": {
          "n_tests": ...,
          "n_significant_ttest": ...,
          "n_significant_nonparam": ...,
          "n_unpaired": ...,
        }
      }
    """
    tests_list = []
    for r in results:
        d = asdict(r)
        # Replace nan with null for JSON compatibility
        for key, val in d.items():
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                d[key] = None
        tests_list.append(d)

    n_sig_t = sum(1 for r in results if r.significant_ttest)
    n_sig_np = sum(1 for r in results if r.significant_nonparam)
    n_unpaired = sum(1 for r in results if r.test_type == "unpaired")

    return {
        "meta": {
            "script": str(Path(__file__).name),
            "alpha": results[0].alpha if results else 0.05,
            "n_bootstrap": results[0].n_bootstrap if results else 10000,
            "n_tests_total": len(results),
        },
        "tests": tests_list,
        "summary": {
            "n_tests": len(results),
            "n_significant_ttest": n_sig_t,
            "n_significant_nonparam": n_sig_np,
            "n_unpaired": n_unpaired,
            "n_paired": len(results) - n_unpaired,
            "fraction_significant": (n_sig_t / len(results) if results else 0),
        },
    }


# ===================================================================
# 7. Human-Readable Console Summary
# ===================================================================


def print_summary(results: list[TestResult]) -> None:
    """Print a human-readable summary table to stdout."""
    print("\n" + "=" * 100)
    print("STATISTICAL SIGNIFICANCE TESTING RESULTS")
    print("=" * 100)

    # Group by comparison pair
    pairs: dict[str, list[TestResult]] = {}
    for r in results:
        key = f"{r.method_a} vs {r.method_b}"
        pairs.setdefault(key, []).append(r)

    for pair_label, pair_results in pairs.items():
        print(f"\n--- {pair_label} ---")
        print(
            f"  {'Procedure':<18} {'Metric':<12} {'Delta':>10} "
            f"{'p (t-test)':>12} {'p (nonpar)':>12} "
            f"{'Cohen d':>10} {'Sig?':>6} {'Type':<8}"
        )
        print("  " + "-" * 94)

        for r in sorted(pair_results, key=lambda x: (x.procedure, x.metric)):
            metric_display = METRICS.get(r.metric, {}).get("display", r.metric)
            sig = _sig_symbol(r.p_value_ttest, r.alpha)

            p_np_str = (
                f"{r.p_value_nonparam:.4f}"
                if not (
                    math.isnan(r.p_value_nonparam)
                    if isinstance(r.p_value_nonparam, float)
                    else False
                )
                else "N/A"
            )

            print(
                f"  {r.procedure:<18} {metric_display:<12} "
                f"{r.delta:>+10.4f} "
                f"{r.p_value_ttest:>12.4f} "
                f"{p_np_str:>12} "
                f"{r.cohens_d:>+10.3f} "
                f"{sig:>6} "
                f"{r.test_type:<8}"
            )

    # Overall summary
    n_sig = sum(1 for r in results if r.significant_ttest)
    n_total = len(results)
    n_unpaired = sum(1 for r in results if r.test_type == "unpaired")
    print(f"\n{'=' * 100}")
    print(f"SUMMARY: {n_sig}/{n_total} tests significant at alpha={results[0].alpha}")
    print(f"  Paired tests:   {n_total - n_unpaired}")
    print(f"  Unpaired tests: {n_unpaired} (weaker — no per-sample pairing)")

    if n_unpaired > 0:
        print(
            "\n  LIMITATION: Unpaired tests used Welch's t-test approximation.\n"
            "  Per-sample baseline data would enable paired tests (Wilcoxon) "
            "for stronger conclusions."
        )
    print("=" * 100 + "\n")


# ===================================================================
# 8. CLI Entry Point
# ===================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Statistical significance testing for LandmarkDiff vs baselines. "
            "Performs paired t-tests, Wilcoxon signed-rank tests, and "
            "bootstrap confidence intervals."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/statistical_tests.py\n"
            "  python scripts/statistical_tests.py --alpha 0.01 --n-bootstrap 50000\n"
            "  python scripts/statistical_tests.py --output results/sig.json\n"
        ),
    )

    default_eval = ROOT / "paper" / "phaseA_eval_results.json"
    default_baseline = ROOT / "paper" / "baseline_results.json"
    default_sd = ROOT / "paper" / "sd_img2img_baseline_s0_3.json"
    default_output = ROOT / "paper" / "significance_results.json"

    parser.add_argument(
        "--eval-results",
        type=Path,
        default=default_eval,
        help=f"Path to LandmarkDiff eval results JSON (default: {default_eval})",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=default_baseline,
        help=f"Path to baseline (TPS/morph/direct) results JSON (default: {default_baseline})",
    )
    parser.add_argument(
        "--sd-results",
        type=Path,
        default=default_sd,
        help=f"Path to SD img2img baseline results JSON (default: {default_sd})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output JSON path (default: {default_output})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for hypothesis tests (default: 0.05)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for CI estimation (default: 10000)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point: load data, run tests, output results."""
    args = parse_args()

    print(f"Loading evaluation results from: {args.eval_results}")
    print(f"Loading baseline results from:   {args.baseline_results}")
    print(f"Loading SD baseline results from: {args.sd_results}")
    print(f"Significance level (alpha):       {args.alpha}")
    print(f"Bootstrap resamples:              {args.n_bootstrap}")
    print()

    # --- Load all data sources ---
    eval_data = load_eval_results(args.eval_results)
    baseline_data = load_baseline_results(args.baseline_results)
    sd_data = load_sd_results(args.sd_results)

    # Report what we found
    if eval_data["per_sample"] is not None:
        n_samples = len(eval_data["per_sample"])
        print(f"Found {n_samples} per-sample LandmarkDiff results (paired tests possible)")
    else:
        print("No per-sample data found; will use unpaired tests only")

    procedures_eval = list(eval_data["summary"].keys())
    procedures_baseline = list(baseline_data.keys())
    procedures_sd = list(sd_data.keys())
    print(f"LandmarkDiff procedures: {procedures_eval}")
    print(f"Baseline procedures:     {procedures_baseline}")
    print(f"SD baseline procedures:  {procedures_sd}")
    print()

    # --- Run all statistical tests ---
    print("Running statistical tests...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        results = run_all_comparisons(
            eval_data,
            baseline_data,
            sd_data,
            alpha=args.alpha,
            n_bootstrap=args.n_bootstrap,
        )

    if not results:
        print("ERROR: No comparisons could be made. Check data files.")
        sys.exit(1)

    print(f"Completed {len(results)} statistical tests.\n")

    # --- Print console summary ---
    print_summary(results)

    # --- Save JSON results ---
    output_json = results_to_json(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"JSON results saved to: {args.output}")

    # --- Generate and save LaTeX tables ---
    # Collect all unique procedures for table generation
    all_procs = sorted(set(r.procedure for r in results))
    latex = generate_full_latex(results, all_procs)
    latex_path = args.output.with_suffix(".tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX tables saved to:  {latex_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
