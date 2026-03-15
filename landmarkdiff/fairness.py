"""Fairness metrics for tracking demographic parity across skin types.

Computes per-Fitzpatrick-type quality metrics and flags regressions
when any group's performance drops below threshold relative to the best
group. Designed for CI integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Fitzpatrick types in order
FITZPATRICK_TYPES = ["I", "II", "III", "IV", "V", "VI"]

# Maximum allowed gap between best and worst group (relative)
_DEFAULT_MAX_GAP = 0.15  # 15% relative gap


@dataclass
class GroupMetrics:
    """Quality metrics for a single demographic group."""

    fitzpatrick_type: str
    count: int = 0
    mean_lpips: float = 0.0
    mean_ssim: float = 0.0
    mean_nme: float = 0.0
    mean_identity_score: float = 0.0

    @property
    def quality_score(self) -> float:
        """Composite quality score (higher = better). Range roughly 0-1."""
        # Weight SSIM and identity positively, LPIPS and NME negatively
        return (
            0.3 * self.mean_ssim
            + 0.3 * self.mean_identity_score
            + 0.2 * (1.0 - min(self.mean_lpips, 1.0))
            + 0.2 * (1.0 - min(self.mean_nme * 10, 1.0))
        )


@dataclass
class FairnessReport:
    """Report on demographic fairness across Fitzpatrick types."""

    groups: dict[str, GroupMetrics] = field(default_factory=dict)
    max_gap: float = _DEFAULT_MAX_GAP
    timestamp: str = ""

    @property
    def best_quality(self) -> float:
        """Highest quality score across all groups."""
        if not self.groups:
            return 0.0
        return max(g.quality_score for g in self.groups.values() if g.count > 0)

    @property
    def worst_quality(self) -> float:
        """Lowest quality score across all groups."""
        active = [g.quality_score for g in self.groups.values() if g.count > 0]
        return min(active) if active else 0.0

    @property
    def quality_gap(self) -> float:
        """Relative gap between best and worst groups."""
        best = self.best_quality
        if best < 1e-6:
            return 0.0
        return (best - self.worst_quality) / best

    @property
    def is_fair(self) -> bool:
        """Whether the quality gap is within acceptable bounds."""
        return self.quality_gap <= self.max_gap

    @property
    def worst_group(self) -> str:
        """Fitzpatrick type with the lowest quality score."""
        active = {k: g.quality_score for k, g in self.groups.items() if g.count > 0}
        if not active:
            return "N/A"
        return min(active, key=active.get)  # type: ignore[arg-type]

    def summary(self) -> str:
        """Human-readable fairness summary."""
        lines = ["Fairness Report"]
        lines.append(f"  Fair: {'YES' if self.is_fair else 'NO'} (gap={self.quality_gap:.1%})")
        lines.append(f"  Threshold: {self.max_gap:.0%}")
        for ft in FITZPATRICK_TYPES:
            g = self.groups.get(ft)
            if g and g.count > 0:
                lines.append(
                    f"  Type {ft}: n={g.count}, quality={g.quality_score:.3f}, "
                    f"LPIPS={g.mean_lpips:.3f}, SSIM={g.mean_ssim:.3f}"
                )
        if not self.is_fair:
            lines.append(f"  WARNING: Type {self.worst_group} underperforms")
        return "\n".join(lines)


def compute_fairness_report(
    results: list[dict],
    max_gap: float = _DEFAULT_MAX_GAP,
) -> FairnessReport:
    """Compute fairness metrics from a list of per-image evaluation results.

    Each result dict should contain:
    - "fitzpatrick": str (Fitzpatrick type I-VI)
    - "lpips": float
    - "ssim": float
    - "nme": float (optional)
    - "identity_score": float (optional)

    Args:
        results: List of per-image evaluation dicts.
        max_gap: Maximum allowed relative quality gap.

    Returns:
        FairnessReport with per-group metrics.
    """
    # Accumulate per-group
    accum: dict[str, list[dict]] = {ft: [] for ft in FITZPATRICK_TYPES}
    for r in results:
        ft = r.get("fitzpatrick", "")
        if ft in accum:
            accum[ft].append(r)

    report = FairnessReport(max_gap=max_gap)
    for ft in FITZPATRICK_TYPES:
        items = accum[ft]
        if not items:
            report.groups[ft] = GroupMetrics(fitzpatrick_type=ft, count=0)
            continue

        report.groups[ft] = GroupMetrics(
            fitzpatrick_type=ft,
            count=len(items),
            mean_lpips=float(np.mean([r.get("lpips", 0.0) for r in items])),
            mean_ssim=float(np.mean([r.get("ssim", 0.0) for r in items])),
            mean_nme=float(np.mean([r.get("nme", 0.0) for r in items])),
            mean_identity_score=float(np.mean([r.get("identity_score", 0.0) for r in items])),
        )

    return report


def check_fairness_regression(
    current: FairnessReport,
    baseline: FairnessReport,
    tolerance: float = 0.05,
) -> list[str]:
    """Check for fairness regressions compared to a baseline.

    Args:
        current: Current evaluation report.
        baseline: Previous baseline report.
        tolerance: Allowed quality drop per group before flagging.

    Returns:
        List of warning messages for regressed groups. Empty = no regressions.
    """
    warnings = []
    for ft in FITZPATRICK_TYPES:
        curr_g = current.groups.get(ft)
        base_g = baseline.groups.get(ft)
        if not curr_g or not base_g or curr_g.count == 0 or base_g.count == 0:
            continue

        drop = base_g.quality_score - curr_g.quality_score
        if drop > tolerance:
            warnings.append(
                f"Type {ft} regressed: quality {base_g.quality_score:.3f} -> "
                f"{curr_g.quality_score:.3f} (drop={drop:.3f})"
            )

    return warnings
