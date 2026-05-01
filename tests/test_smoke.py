from agent_safety_eval_lab.mock_runner import run_demo


def test_mock_eval_runs():
    summary, lines = run_demo()
    assert summary["cases"] == 3
    assert summary["passed"] == 2
    assert any("tool_policy_violation" in line for line in lines)
