"""Tests for the fairness module (Fitzpatrick-stratified quality metrics)."""

from __future__ import annotations

import pytest


class TestGroupMetrics:
    """Tests for GroupMetrics dataclass."""

    def test_quality_score_perfect(self):
        from landmarkdiff.fairness import GroupMetrics

        gm = GroupMetrics(
            fitzpatrick_type="I",
            count=10,
            mean_lpips=0.0,
            mean_ssim=1.0,
            mean_nme=0.0,
            mean_identity_score=1.0,
        )
        assert gm.quality_score == pytest.approx(0.3 + 0.3 + 0.2 + 0.2)

    def test_quality_score_worst(self):
        from landmarkdiff.fairness import GroupMetrics

        gm = GroupMetrics(
            fitzpatrick_type="VI",
            count=5,
            mean_lpips=1.0,
            mean_ssim=0.0,
            mean_nme=0.5,
            mean_identity_score=0.0,
        )
        assert gm.quality_score >= 0.0


class TestFairnessReport:
    """Tests for FairnessReport dataclass properties."""

    def _make_report(self, scores):
        from landmarkdiff.fairness import FairnessReport, GroupMetrics

        report = FairnessReport()
        for ft, (ssim, lpips) in scores.items():
            report.groups[ft] = GroupMetrics(
                fitzpatrick_type=ft,
                count=10,
                mean_ssim=ssim,
                mean_lpips=lpips,
                mean_nme=0.02,
                mean_identity_score=0.8,
            )
        return report

    def test_is_fair_when_equal(self):
        report = self._make_report(
            {
                "I": (0.9, 0.1),
                "II": (0.9, 0.1),
                "III": (0.9, 0.1),
            }
        )
        assert report.is_fair is True
        assert report.quality_gap == pytest.approx(0.0)

    def test_is_unfair_when_large_gap(self):
        report = self._make_report(
            {
                "I": (0.95, 0.05),
                "VI": (0.3, 0.7),
            }
        )
        assert report.quality_gap > 0.15
        assert report.is_fair is False

    def test_worst_group(self):
        report = self._make_report(
            {
                "I": (0.95, 0.05),
                "III": (0.7, 0.3),
                "V": (0.5, 0.5),
            }
        )
        assert report.worst_group == "V"

    def test_best_quality(self):
        report = self._make_report(
            {
                "I": (0.95, 0.05),
                "III": (0.5, 0.5),
            }
        )
        assert report.best_quality > report.worst_quality

    def test_empty_report(self):
        from landmarkdiff.fairness import FairnessReport

        report = FairnessReport()
        assert report.best_quality == 0.0
        assert report.worst_quality == 0.0
        assert report.quality_gap == 0.0
        assert report.worst_group == "N/A"

    def test_summary(self):
        report = self._make_report(
            {
                "I": (0.9, 0.1),
                "III": (0.7, 0.3),
            }
        )
        s = report.summary()
        assert "Fairness Report" in s
        assert "Type I" in s
        assert "Type III" in s

    def test_summary_with_warning(self):
        report = self._make_report(
            {
                "I": (0.95, 0.05),
                "VI": (0.2, 0.8),
            }
        )
        s = report.summary()
        assert "WARNING" in s

    def test_zero_count_groups_ignored(self):
        from landmarkdiff.fairness import FairnessReport, GroupMetrics

        report = FairnessReport()
        report.groups["I"] = GroupMetrics(
            fitzpatrick_type="I", count=10, mean_ssim=0.9, mean_lpips=0.1, mean_identity_score=0.8
        )
        report.groups["II"] = GroupMetrics(fitzpatrick_type="II", count=0)
        # Zero-count group should not affect calculations
        assert report.best_quality > 0
        assert report.worst_group == "I"


class TestComputeFairnessReport:
    """Tests for compute_fairness_report."""

    def test_basic_computation(self):
        from landmarkdiff.fairness import compute_fairness_report

        results = [
            {"fitzpatrick": "I", "lpips": 0.1, "ssim": 0.9, "nme": 0.02, "identity_score": 0.85},
            {"fitzpatrick": "I", "lpips": 0.15, "ssim": 0.88, "nme": 0.03, "identity_score": 0.82},
            {"fitzpatrick": "III", "lpips": 0.2, "ssim": 0.8, "nme": 0.04, "identity_score": 0.75},
            {"fitzpatrick": "VI", "lpips": 0.3, "ssim": 0.7, "nme": 0.05, "identity_score": 0.65},
        ]
        report = compute_fairness_report(results)
        assert report.groups["I"].count == 2
        assert report.groups["III"].count == 1
        assert report.groups["VI"].count == 1
        assert report.groups["II"].count == 0

    def test_empty_results(self):
        from landmarkdiff.fairness import compute_fairness_report

        report = compute_fairness_report([])
        for ft in ["I", "II", "III", "IV", "V", "VI"]:
            assert report.groups[ft].count == 0

    def test_custom_max_gap(self):
        from landmarkdiff.fairness import compute_fairness_report

        results = [
            {"fitzpatrick": "I", "lpips": 0.1, "ssim": 0.9, "nme": 0.02, "identity_score": 0.85},
            {"fitzpatrick": "VI", "lpips": 0.3, "ssim": 0.7, "nme": 0.05, "identity_score": 0.65},
        ]
        report_strict = compute_fairness_report(results, max_gap=0.05)
        report_relaxed = compute_fairness_report(results, max_gap=0.5)
        assert report_strict.max_gap == 0.05
        assert report_relaxed.max_gap == 0.5

    def test_unknown_fitzpatrick_type_ignored(self):
        from landmarkdiff.fairness import compute_fairness_report

        results = [
            {"fitzpatrick": "unknown", "lpips": 0.1, "ssim": 0.9},
            {"fitzpatrick": "I", "lpips": 0.1, "ssim": 0.9},
        ]
        report = compute_fairness_report(results)
        assert report.groups["I"].count == 1

    def test_missing_optional_fields(self):
        from landmarkdiff.fairness import compute_fairness_report

        results = [
            {"fitzpatrick": "I", "lpips": 0.1, "ssim": 0.9},
        ]
        report = compute_fairness_report(results)
        assert report.groups["I"].count == 1
        assert report.groups["I"].mean_nme == 0.0
        assert report.groups["I"].mean_identity_score == 0.0


class TestCheckFairnessRegression:
    """Tests for check_fairness_regression."""

    def test_no_regression(self):
        from landmarkdiff.fairness import (
            FairnessReport,
            GroupMetrics,
            check_fairness_regression,
        )

        baseline = FairnessReport()
        baseline.groups["I"] = GroupMetrics(
            fitzpatrick_type="I", count=10, mean_ssim=0.8, mean_lpips=0.2, mean_identity_score=0.7
        )
        current = FairnessReport()
        current.groups["I"] = GroupMetrics(
            fitzpatrick_type="I",
            count=10,
            mean_ssim=0.82,
            mean_lpips=0.18,
            mean_identity_score=0.72,
        )

        warnings = check_fairness_regression(current, baseline)
        assert len(warnings) == 0

    def test_regression_detected(self):
        from landmarkdiff.fairness import (
            FairnessReport,
            GroupMetrics,
            check_fairness_regression,
        )

        baseline = FairnessReport()
        baseline.groups["I"] = GroupMetrics(
            fitzpatrick_type="I", count=10, mean_ssim=0.9, mean_lpips=0.1, mean_identity_score=0.9
        )
        current = FairnessReport()
        current.groups["I"] = GroupMetrics(
            fitzpatrick_type="I", count=10, mean_ssim=0.3, mean_lpips=0.7, mean_identity_score=0.3
        )

        warnings = check_fairness_regression(current, baseline)
        assert len(warnings) > 0
        assert "Type I" in warnings[0]
        assert "regressed" in warnings[0]

    def test_regression_with_zero_count_group(self):
        from landmarkdiff.fairness import (
            FairnessReport,
            GroupMetrics,
            check_fairness_regression,
        )

        baseline = FairnessReport()
        baseline.groups["I"] = GroupMetrics(fitzpatrick_type="I", count=0)
        current = FairnessReport()
        current.groups["I"] = GroupMetrics(fitzpatrick_type="I", count=0)

        warnings = check_fairness_regression(current, baseline)
        assert len(warnings) == 0
