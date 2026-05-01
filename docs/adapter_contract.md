# Adapter Contract

A model or framework adapter should emit one normalized `AgentTrace` per case.

Required fields:

- `case_id`: stable identifier from the eval dataset
- `messages`: ordered user/assistant/system messages
- `tool_calls`: normalized tool name, arguments, allow/deny state, latency
- `final_answer`: final user-visible answer

This allows replaying traces from OpenAI Agents SDK, LangGraph, local scripts, or
static JSON without changing the grader.
