from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch


@dataclass(frozen=True)
class PolicyRule:
    tool: str
    decision: str
    reason: str
    path_globs: tuple[str, ...] = ()
    max_calls: int | None = None


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    matched_rule: str


class ToolPolicy:
    def __init__(self, rules: list[PolicyRule]) -> None:
        self.rules = rules

    def decide(self, tool: str, arguments: dict[str, str], prior_calls: int = 0) -> PolicyDecision:
        for rule in self.rules:
            if rule.tool != tool:
                continue
            if rule.max_calls is not None and prior_calls >= rule.max_calls:
                return PolicyDecision(False, f"call budget exceeded: {rule.reason}", rule.tool)
            path = arguments.get("path")
            if rule.path_globs and path and not any(fnmatch(path, glob) for glob in rule.path_globs):
                return PolicyDecision(False, f"path outside policy: {path}", rule.tool)
            return PolicyDecision(rule.decision == "allow", rule.reason, rule.tool)
        return PolicyDecision(False, "tool not declared in policy", "*")


DEFAULT_POLICY = ToolPolicy(
    [
        PolicyRule("search.docs", "allow", "read-only retrieval", max_calls=4),
        PolicyRule("calculator", "allow", "deterministic computation", max_calls=3),
        PolicyRule("file.read", "allow", "public workspace reads only", path_globs=("workspace/public/*",), max_calls=2),
        PolicyRule("file.delete", "deny", "destructive file mutation"),
        PolicyRule("network.post", "deny", "external data egress"),
        PolicyRule("shell.exec", "deny", "arbitrary execution surface"),
    ]
)
