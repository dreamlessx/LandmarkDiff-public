"""Clinical audit report generator for regulatory compliance.

Generates structured HTML reports summarizing safety validation results,
model performance, and Fitzpatrick equity analysis for clinical review.

Reports include:
- Safety validation pass/fail summary per patient
- Aggregate statistics by procedure and Fitzpatrick type
- Flagged cases for manual review
- Model version and configuration provenance

Usage:
    from landmarkdiff.audit import AuditReporter, AuditCase

    reporter = AuditReporter(model_version="0.3.0")
    reporter.add_case(AuditCase(
        case_id="P001",
        procedure="rhinoplasty",
        safety_passed=True,
        identity_sim=0.87,
        fitzpatrick_type="III",
    ))
    reporter.generate_report("audit_report.html")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AuditCase:
    """A single patient case for audit reporting."""

    case_id: str
    procedure: str
    safety_passed: bool
    identity_sim: float = 0.0
    intensity: float = 65.0
    fitzpatrick_type: str = ""
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AuditSummary:
    """Aggregate statistics for an audit report."""

    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    flagged_cases: int = 0
    pass_rate: float = 0.0
    mean_identity_sim: float = 0.0
    by_procedure: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_fitzpatrick: dict[str, dict[str, Any]] = field(default_factory=dict)


class AuditReporter:
    """Generate clinical audit reports from safety validation results.

    Args:
        model_version: Model version string for provenance.
        report_title: Title for generated reports.
    """

    def __init__(
        self,
        model_version: str = "0.3.0",
        report_title: str = "LandmarkDiff Clinical Audit Report",
    ) -> None:
        self.model_version = model_version
        self.report_title = report_title
        self.cases: list[AuditCase] = []

    def add_case(self, case: AuditCase) -> None:
        """Add a case to the audit report."""
        self.cases.append(case)

    def add_cases(self, cases: list[AuditCase]) -> None:
        """Add multiple cases."""
        self.cases.extend(cases)

    def clear(self) -> None:
        """Clear all cases."""
        self.cases.clear()

    def compute_summary(self) -> AuditSummary:
        """Compute aggregate statistics from all cases."""
        if not self.cases:
            return AuditSummary()

        total = len(self.cases)
        passed = sum(1 for c in self.cases if c.safety_passed)
        failed = total - passed
        flagged = sum(1 for c in self.cases if not c.safety_passed or c.warnings)

        id_sims = [c.identity_sim for c in self.cases if c.identity_sim > 0]
        mean_id = sum(id_sims) / len(id_sims) if id_sims else 0.0

        # By procedure
        by_proc: dict[str, dict[str, Any]] = {}
        for case in self.cases:
            proc = case.procedure
            if proc not in by_proc:
                by_proc[proc] = {"total": 0, "passed": 0, "id_sims": []}
            by_proc[proc]["total"] += 1
            if case.safety_passed:
                by_proc[proc]["passed"] += 1
            if case.identity_sim > 0:
                by_proc[proc]["id_sims"].append(case.identity_sim)

        for proc, stats in by_proc.items():
            stats["pass_rate"] = stats["passed"] / max(stats["total"], 1)
            stats["mean_identity_sim"] = (
                sum(stats["id_sims"]) / len(stats["id_sims"])
                if stats["id_sims"]
                else 0.0
            )
            del stats["id_sims"]

        # By Fitzpatrick type
        by_fitz: dict[str, dict[str, Any]] = {}
        for case in self.cases:
            ft = case.fitzpatrick_type or "Unknown"
            if ft not in by_fitz:
                by_fitz[ft] = {"total": 0, "passed": 0, "id_sims": []}
            by_fitz[ft]["total"] += 1
            if case.safety_passed:
                by_fitz[ft]["passed"] += 1
            if case.identity_sim > 0:
                by_fitz[ft]["id_sims"].append(case.identity_sim)

        for ft, stats in by_fitz.items():
            stats["pass_rate"] = stats["passed"] / max(stats["total"], 1)
            stats["mean_identity_sim"] = (
                sum(stats["id_sims"]) / len(stats["id_sims"])
                if stats["id_sims"]
                else 0.0
            )
            del stats["id_sims"]

        return AuditSummary(
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            flagged_cases=flagged,
            pass_rate=passed / total,
            mean_identity_sim=mean_id,
            by_procedure=by_proc,
            by_fitzpatrick=by_fitz,
        )

    def flagged_cases(self) -> list[AuditCase]:
        """Return cases that need manual review (failed or have warnings)."""
        return [c for c in self.cases if not c.safety_passed or c.warnings]

    def to_json(self) -> str:
        """Export audit data as JSON."""
        summary = self.compute_summary()
        data = {
            "report_title": self.report_title,
            "model_version": self.model_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_cases": summary.total_cases,
                "passed_cases": summary.passed_cases,
                "failed_cases": summary.failed_cases,
                "flagged_cases": summary.flagged_cases,
                "pass_rate": round(summary.pass_rate, 4),
                "mean_identity_sim": round(summary.mean_identity_sim, 4),
            },
            "by_procedure": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in summary.by_procedure.items()
            },
            "by_fitzpatrick": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in summary.by_fitzpatrick.items()
            },
            "cases": [
                {
                    "case_id": c.case_id,
                    "procedure": c.procedure,
                    "safety_passed": c.safety_passed,
                    "identity_sim": round(c.identity_sim, 4),
                    "intensity": c.intensity,
                    "fitzpatrick_type": c.fitzpatrick_type,
                    "warnings": c.warnings,
                    "failures": c.failures,
                    "metrics": {k: round(v, 4) for k, v in c.metrics.items()},
                    "timestamp": c.timestamp,
                }
                for c in self.cases
            ],
        }
        return json.dumps(data, indent=2)

    def generate_report(self, output_path: str | Path) -> Path:
        """Generate an HTML audit report.

        Args:
            output_path: Path to save the HTML report.

        Returns:
            Path to the generated report.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.compute_summary()
        html = self._render_html(summary)

        output_path.write_text(html)
        return output_path

    def _render_html(self, summary: AuditSummary) -> str:
        """Render the audit report as HTML."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        status = "PASS" if summary.failed_cases == 0 else "REQUIRES REVIEW"
        status_color = "#28a745" if summary.failed_cases == 0 else "#dc3545"

        # Build procedure rows
        proc_rows = ""
        for proc, stats in sorted(summary.by_procedure.items()):
            rate = stats["pass_rate"]
            rate_color = "#28a745" if rate >= 0.95 else "#ffc107" if rate >= 0.8 else "#dc3545"
            proc_rows += (
                f"<tr>"
                f"<td>{proc.title()}</td>"
                f"<td>{stats['total']}</td>"
                f"<td>{stats['passed']}</td>"
                f'<td style="color:{rate_color};font-weight:bold">{rate:.1%}</td>'
                f"<td>{stats['mean_identity_sim']:.4f}</td>"
                f"</tr>\n"
            )

        # Build Fitzpatrick rows
        fitz_rows = ""
        for ft, stats in sorted(summary.by_fitzpatrick.items()):
            rate = stats["pass_rate"]
            rate_color = "#28a745" if rate >= 0.95 else "#ffc107" if rate >= 0.8 else "#dc3545"
            fitz_rows += (
                f"<tr>"
                f"<td>{ft}</td>"
                f"<td>{stats['total']}</td>"
                f"<td>{stats['passed']}</td>"
                f'<td style="color:{rate_color};font-weight:bold">{rate:.1%}</td>'
                f"<td>{stats['mean_identity_sim']:.4f}</td>"
                f"</tr>\n"
            )

        # Build flagged cases
        flagged = self.flagged_cases()
        flagged_rows = ""
        for c in flagged:
            issues = "; ".join(c.failures + [f"WARN: {w}" for w in c.warnings])
            bg = "#fff3cd" if c.safety_passed else "#f8d7da"
            flagged_rows += (
                f'<tr style="background:{bg}">'
                f"<td>{c.case_id}</td>"
                f"<td>{c.procedure.title()}</td>"
                f"<td>{c.fitzpatrick_type}</td>"
                f"<td>{c.identity_sim:.4f}</td>"
                f'<td>{"WARN" if c.safety_passed else "FAIL"}</td>'
                f"<td>{issues}</td>"
                f"</tr>\n"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{self.report_title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 20px; color: #333; }}
h1 {{ border-bottom: 3px solid #333; padding-bottom: 10px; }}
h2 {{ color: #555; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #f8f9fa; font-weight: 600; }}
tr:hover {{ background: #f5f5f5; }}
.status {{ display: inline-block; padding: 4px 12px; border-radius: 4px;
           color: white; font-weight: bold; font-size: 18px; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                 gap: 15px; margin: 20px 0; }}
.summary-card {{ background: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center; }}
.summary-card .value {{ font-size: 28px; font-weight: bold; color: #333; }}
.summary-card .label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
.disclaimer {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;
               padding: 12px; margin: 20px 0; font-size: 13px; }}
footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd;
          font-size: 12px; color: #999; }}
</style>
</head>
<body>
<h1>{self.report_title}</h1>
<p>Generated: {now} &nbsp;|&nbsp; Model version: <code>{self.model_version}</code>
&nbsp;|&nbsp; Overall status: <span class="status" style="background:{status_color}">{status}</span></p>

<div class="disclaimer">
<strong>Disclaimer:</strong> This report is for research and development purposes only.
LandmarkDiff predictions are AI-generated visualizations and do not constitute medical advice
or guarantee surgical outcomes. All predictions should be reviewed by qualified clinical professionals.
</div>

<h2>Summary</h2>
<div class="summary-grid">
<div class="summary-card"><div class="value">{summary.total_cases}</div><div class="label">Total Cases</div></div>
<div class="summary-card"><div class="value" style="color:#28a745">{summary.passed_cases}</div><div class="label">Passed</div></div>
<div class="summary-card"><div class="value" style="color:#dc3545">{summary.failed_cases}</div><div class="label">Failed</div></div>
<div class="summary-card"><div class="value" style="color:#ffc107">{summary.flagged_cases}</div><div class="label">Flagged</div></div>
<div class="summary-card"><div class="value">{summary.pass_rate:.1%}</div><div class="label">Pass Rate</div></div>
<div class="summary-card"><div class="value">{summary.mean_identity_sim:.4f}</div><div class="label">Mean ID Sim</div></div>
</div>

<h2>Performance by Procedure</h2>
<table>
<tr><th>Procedure</th><th>Total</th><th>Passed</th><th>Pass Rate</th><th>Mean ID Sim</th></tr>
{proc_rows}</table>

<h2>Equity Analysis by Fitzpatrick Type</h2>
<table>
<tr><th>Fitzpatrick Type</th><th>Total</th><th>Passed</th><th>Pass Rate</th><th>Mean ID Sim</th></tr>
{fitz_rows}</table>

{"<h2>Flagged Cases (Require Review)</h2>" if flagged_rows else ""}
{"<table><tr><th>Case ID</th><th>Procedure</th><th>Fitzpatrick</th><th>ID Sim</th><th>Status</th><th>Issues</th></tr>" + flagged_rows + "</table>" if flagged_rows else "<p>No flagged cases.</p>"}

<footer>
LandmarkDiff v{self.model_version} &mdash; Clinical Audit Report &mdash;
For research use only. Not FDA approved.
</footer>
</body>
</html>"""
