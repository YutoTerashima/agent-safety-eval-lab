from __future__ import annotations

from collections import Counter


REQUIRED_RISKS = {"pass", "tool_policy_violation", "unsafe_content", "tool_budget_violation"}


def suite_coverage(cases: list[dict]) -> dict[str, object]:
    risks = Counter(case["expected_risk"] for case in cases)
    tools = Counter(tool for case in cases for tool in case.get("tool_sequence", []))
    missing = sorted(REQUIRED_RISKS - set(risks))
    return {
        "cases": len(cases),
        "risk_coverage": dict(sorted(risks.items())),
        "tool_coverage": dict(sorted(tools.items())),
        "missing_required_risks": missing,
        "coverage_passed": not missing and len(tools) >= 5,
    }
