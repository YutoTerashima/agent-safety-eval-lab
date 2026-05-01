# Architecture Notes

The core abstraction is `AgentTrace`: normalized messages plus tool calls and a
final answer. Evaluation is split into two independent passes:

1. Tool policy grading checks whether each tool call is inside the case policy.
2. Safety rubric grading checks whether the final answer contains case-specific
   blocked terms or requires human review.

This keeps model adapters replaceable and makes replay-based debugging possible.
