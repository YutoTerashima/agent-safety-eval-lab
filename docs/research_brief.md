# Research Brief

## Problem

Final-answer evaluation misses failures that happen inside an agent trajectory:
unnecessary tool calls, denied tool calls, unsafe intermediate state, and policy
violations that are later hidden by a polished final answer.

## Method

The lab evaluates normalized traces with two independent passes:

1. **Tool policy grading** checks whether each tool call is permitted for the case.
2. **Safety rubric grading** checks final-answer behavior and case-specific blocked terms.

The separation matters because tool risk and language risk fail differently.

## What the Mock Demo Proves

The destructive file request fails even though the final answer refuses safely. This
demonstrates why trace-level eval is stricter than output-only eval.

## Limitations

- The demo uses deterministic mock traces.
- The rubric is intentionally simple.
- No private prompts, hidden model outputs, or internal systems are included.

## Next Experiments

- Add replay support for JSONL traces from real agent frameworks.
- Add latency and tool-count efficiency metrics.
- Add human-review labels for ambiguous cases.
