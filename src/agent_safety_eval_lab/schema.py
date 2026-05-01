from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, str]
    allowed: bool
    latency_ms: int = 0


@dataclass(frozen=True)
class AgentTrace:
    case_id: str
    user_goal: str
    messages: list[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str = ""


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    user_goal: str
    allowed_tools: set[str]
    blocked_terms: set[str]


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    passed: bool
    risk_category: str
    findings: list[str]
    trace_steps: int
