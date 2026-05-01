from __future__ import annotations

import json
from pathlib import Path

from .grader import grade_trace, summarize
from .schema import AgentTrace, EvalCase, ToolCall


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            case_id=item["case_id"],
            user_goal=item["user_goal"],
            allowed_tools=set(item["allowed_tools"]),
            blocked_terms=set(item["blocked_terms"]),
        )
        for item in raw
    ]


def mock_agent(case: EvalCase) -> AgentTrace:
    if "delete" in case.user_goal.lower():
        calls = [ToolCall("file.delete", {"path": "workspace/secrets.txt"}, allowed=False, latency_ms=9)]
        answer = "I cannot delete files outside the allowed workflow. I can produce a review plan instead."
    elif "summarize" in case.user_goal.lower():
        calls = [ToolCall("search.docs", {"query": "agent evaluation"}, allowed=True, latency_ms=12)]
        answer = "The requested material describes trace-based evaluation and tool-call review."
    else:
        calls = [ToolCall("calculator", {"expression": "2+2"}, allowed=True, latency_ms=2)]
        answer = "The mock agent completed the task without unsafe content."

    return AgentTrace(
        case_id=case.case_id,
        user_goal=case.user_goal,
        messages=[f"user:{case.user_goal}", "assistant:planning", "assistant:final"],
        tool_calls=calls,
        final_answer=answer,
    )


def run_demo(dataset_path: Path | None = None) -> tuple[dict[str, int], list[str]]:
    if dataset_path is None:
        dataset_path = Path(__file__).resolve().parents[2] / "datasets" / "mock_agent_cases.json"
    cases = load_cases(dataset_path)
    results = [grade_trace(mock_agent(case), case) for case in cases]
    lines = [
        f"{result.case_id}: {'pass' if result.passed else 'fail'} | {result.risk_category}"
        + (f" | {', '.join(result.findings)}" if result.findings else "")
        for result in results
    ]
    return summarize(results), lines
