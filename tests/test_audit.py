"""Tests for landmarkdiff.audit."""

from __future__ import annotations

import json

import pytest

from landmarkdiff.audit import AuditCase, AuditReporter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_cases():
    return [
        AuditCase(
            case_id="P001",
            procedure="rhinoplasty",
            safety_passed=True,
            identity_sim=0.87,
            fitzpatrick_type="I-II",
        ),
        AuditCase(
            case_id="P002",
            procedure="rhinoplasty",
            safety_passed=True,
            identity_sim=0.91,
            fitzpatrick_type="III-IV",
        ),
        AuditCase(
            case_id="P003",
            procedure="blepharoplasty",
            safety_passed=True,
            identity_sim=0.89,
            fitzpatrick_type="I-II",
        ),
        AuditCase(
            case_id="P004",
            procedure="blepharoplasty",
            safety_passed=False,
            identity_sim=0.45,
            fitzpatrick_type="V-VI",
            failures=["Identity similarity 0.45 below threshold 0.5"],
        ),
        AuditCase(
            case_id="P005",
            procedure="orthognathic",
            safety_passed=True,
            identity_sim=0.82,
            fitzpatrick_type="III-IV",
            warnings=["Unusual aspect ratio: 2.5"],
        ),
    ]


@pytest.fixture
def reporter(sample_cases):
    r = AuditReporter(model_version="0.3.2")
    r.add_cases(sample_cases)
    return r


# ---------------------------------------------------------------------------
# AuditCase
# ---------------------------------------------------------------------------


class TestAuditCase:
    def test_defaults(self):
        case = AuditCase(case_id="X", procedure="test", safety_passed=True)
        assert case.identity_sim == 0.0
        assert case.intensity == 65.0
        assert case.fitzpatrick_type == ""
        assert case.warnings == []
        assert case.failures == []
        assert case.metrics == {}
        assert case.timestamp  # auto-populated

    def test_with_metrics(self):
        case = AuditCase(
            case_id="X",
            procedure="rhinoplasty",
            safety_passed=True,
            metrics={"ssim": 0.89, "lpips": 0.11},
        )
        assert case.metrics["ssim"] == 0.89


# ---------------------------------------------------------------------------
# AuditReporter init
# ---------------------------------------------------------------------------


class TestReporterInit:
    def test_defaults(self):
        r = AuditReporter()
        assert r.model_version == "0.3.2"
        assert r.cases == []

    def test_add_case(self):
        r = AuditReporter()
        r.add_case(AuditCase(case_id="A", procedure="test", safety_passed=True))
        assert len(r.cases) == 1

    def test_add_cases(self, sample_cases):
        r = AuditReporter()
        r.add_cases(sample_cases)
        assert len(r.cases) == 5

    def test_clear(self, reporter):
        assert len(reporter.cases) > 0
        reporter.clear()
        assert len(reporter.cases) == 0


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


class TestComputeSummary:
    def test_counts(self, reporter):
        s = reporter.compute_summary()
        assert s.total_cases == 5
        assert s.passed_cases == 4
        assert s.failed_cases == 1
        assert s.flagged_cases == 2  # P004 (failed) + P005 (warning)

    def test_pass_rate(self, reporter):
        s = reporter.compute_summary()
        assert s.pass_rate == pytest.approx(0.8)

    def test_mean_identity(self, reporter):
        s = reporter.compute_summary()
        expected = (0.87 + 0.91 + 0.89 + 0.45 + 0.82) / 5
        assert s.mean_identity_sim == pytest.approx(expected, rel=1e-3)

    def test_by_procedure(self, reporter):
        s = reporter.compute_summary()
        assert "rhinoplasty" in s.by_procedure
        assert "blepharoplasty" in s.by_procedure
        assert "orthognathic" in s.by_procedure
        assert s.by_procedure["rhinoplasty"]["total"] == 2
        assert s.by_procedure["rhinoplasty"]["passed"] == 2
        assert s.by_procedure["blepharoplasty"]["total"] == 2
        assert s.by_procedure["blepharoplasty"]["passed"] == 1

    def test_by_fitzpatrick(self, reporter):
        s = reporter.compute_summary()
        assert "I-II" in s.by_fitzpatrick
        assert "III-IV" in s.by_fitzpatrick
        assert "V-VI" in s.by_fitzpatrick
        assert s.by_fitzpatrick["I-II"]["total"] == 2
        assert s.by_fitzpatrick["V-VI"]["passed"] == 0

    def test_empty_summary(self):
        r = AuditReporter()
        s = r.compute_summary()
        assert s.total_cases == 0
        assert s.pass_rate == 0.0

    def test_procedure_pass_rate(self, reporter):
        s = reporter.compute_summary()
        assert s.by_procedure["rhinoplasty"]["pass_rate"] == pytest.approx(1.0)
        assert s.by_procedure["blepharoplasty"]["pass_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Flagged cases
# ---------------------------------------------------------------------------


class TestFlaggedCases:
    def test_returns_failed_and_warned(self, reporter):
        flagged = reporter.flagged_cases()
        ids = {c.case_id for c in flagged}
        assert "P004" in ids  # failed
        assert "P005" in ids  # warning
        assert "P001" not in ids  # clean pass

    def test_no_flagged_when_all_pass(self):
        r = AuditReporter()
        r.add_case(AuditCase(case_id="A", procedure="test", safety_passed=True))
        assert len(r.flagged_cases()) == 0


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


class TestToJson:
    def test_valid_json(self, reporter):
        j = reporter.to_json()
        data = json.loads(j)
        assert data["model_version"] == "0.3.2"
        assert len(data["cases"]) == 5

    def test_summary_in_json(self, reporter):
        data = json.loads(reporter.to_json())
        assert data["summary"]["total_cases"] == 5
        assert data["summary"]["pass_rate"] == pytest.approx(0.8)

    def test_by_procedure_in_json(self, reporter):
        data = json.loads(reporter.to_json())
        assert "rhinoplasty" in data["by_procedure"]

    def test_case_fields(self, reporter):
        data = json.loads(reporter.to_json())
        case = data["cases"][0]
        assert "case_id" in case
        assert "procedure" in case
        assert "safety_passed" in case
        assert "identity_sim" in case
        assert "timestamp" in case


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_creates_file(self, reporter, tmp_path):
        out = tmp_path / "report.html"
        result = reporter.generate_report(out)
        assert result.exists()
        assert result.suffix == ".html"

    def test_html_content(self, reporter, tmp_path):
        out = tmp_path / "report.html"
        reporter.generate_report(out)
        html = out.read_text()
        assert "Clinical Audit Report" in html
        assert "0.3.2" in html
        assert "Rhinoplasty" in html
        assert "P004" in html  # flagged case

    def test_creates_parent_dirs(self, reporter, tmp_path):
        out = tmp_path / "sub" / "dir" / "report.html"
        result = reporter.generate_report(out)
        assert result.exists()

    def test_all_pass_no_flagged_section(self, tmp_path):
        r = AuditReporter()
        r.add_case(AuditCase(case_id="A", procedure="test", safety_passed=True, identity_sim=0.9))
        out = tmp_path / "report.html"
        r.generate_report(out)
        html = out.read_text()
        assert "No flagged cases" in html

    def test_disclaimer_present(self, reporter, tmp_path):
        out = tmp_path / "report.html"
        reporter.generate_report(out)
        html = out.read_text()
        assert "research and development purposes only" in html
        assert "Not FDA approved" in html

    def test_equity_table_present(self, reporter, tmp_path):
        out = tmp_path / "report.html"
        reporter.generate_report(out)
        html = out.read_text()
        assert "Fitzpatrick Type" in html
        assert "I-II" in html
        assert "V-VI" in html
