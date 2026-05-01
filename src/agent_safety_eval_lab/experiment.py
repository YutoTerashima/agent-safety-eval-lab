from __future__ import annotations

import json
from pathlib import Path

from .analytics import aggregate_results, analyze_trace
from .grader import grade_trace
from .mock_runner import load_cases, mock_agent


def run_experiment(dataset_path: Path | None = None) -> dict[str, object]:
    if dataset_path is None:
        dataset_path = Path(__file__).resolve().parents[2] / "datasets" / "mock_agent_cases.json"
    cases = load_cases(dataset_path)
    results = []
    trace_analytics = []
    for case in cases:
        trace = mock_agent(case)
        result = grade_trace(trace, case)
        results.append(result)
        trace_analytics.append(analyze_trace(trace, result))

    return {
        "summary": aggregate_results(results),
        "cases": [
            {
                "case_id": result.case_id,
                "passed": result.passed,
                "risk_category": result.risk_category,
                "findings": result.findings,
                "trace_steps": result.trace_steps,
            }
            for result in results
        ],
        "trace_analytics": [item.__dict__ for item in trace_analytics],
    }


def write_experiment_outputs(report_dir: Path | None = None) -> tuple[Path, Path]:
    if report_dir is None:
        report_dir = Path(__file__).resolve().parents[2] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    data = run_experiment()
    json_path = report_dir / "trace_analytics.json"
    md_path = report_dir / "trace_analytics_report.md"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    rows = [
        f"| {item['case_id']} | {item['risk_score']} | {item['tool_count']} | {item['denied_tool_count']} | {item['total_latency_ms']} |"
        for item in data["trace_analytics"]
    ]
    md_path.write_text(
        "\n".join(
            [
                "# Trace Analytics Report",
                "",
                "## Aggregate",
                "",
                f"- Pass rate: {data['summary']['pass_rate']}",
                f"- Review queue: {', '.join(data['summary']['review_queue']) or 'empty'}",
                f"- Risk categories: {data['summary']['risk_categories']}",
                "",
                "## Per-Trace Metrics",
                "",
                "| Case | Risk Score | Tools | Denied Tools | Latency ms |",
                "| --- | ---: | ---: | ---: | ---: |",
                *rows,
                "",
                "The destructive request receives the highest risk score because it combines an unauthorized tool request with a policy denial.",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, md_path
