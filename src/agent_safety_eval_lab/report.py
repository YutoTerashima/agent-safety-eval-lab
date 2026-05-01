from __future__ import annotations

import json
from pathlib import Path

from .mock_runner import run_demo


def build_markdown(summary: dict[str, int], lines: list[str]) -> str:
    rows = []
    for line in lines:
        case_id, rest = line.split(": ", 1)
        parts = [part.strip() for part in rest.split("|")]
        status = parts[0]
        risk = parts[1] if len(parts) > 1 else "unknown"
        findings = parts[2] if len(parts) > 2 else ""
        rows.append(f"| {case_id} | {status} | {risk} | {findings} |")

    return "\n".join(
        [
            "# Demo Evaluation Report",
            "",
            "## Summary",
            "",
            f"- Cases: {summary['cases']}",
            f"- Passed: {summary['passed']}",
            f"- Failed: {summary['failed']}",
            f"- High risk: {summary['high_risk']}",
            "",
            "## Case Results",
            "",
            "| Case | Status | Risk | Findings |",
            "| --- | --- | --- | --- |",
            *rows,
            "",
            "## Interpretation",
            "",
            "The destructive file request is caught as a tool-policy violation before any final-answer-only safety check would notice it.",
        ]
    )


def write_report(report_dir: Path | None = None) -> tuple[Path, Path]:
    if report_dir is None:
        report_dir = Path(__file__).resolve().parents[2] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary, lines = run_demo()
    json_path = report_dir / "demo_results.json"
    md_path = report_dir / "demo_report.md"
    json_path.write_text(json.dumps({"summary": summary, "cases": lines}, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(summary, lines), encoding="utf-8")
    return json_path, md_path
