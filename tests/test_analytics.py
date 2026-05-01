from agent_safety_eval_lab.experiment import run_experiment


def test_experiment_has_review_queue():
    data = run_experiment()
    assert data["summary"]["pass_rate"] < 1
    assert "C-002" in data["summary"]["review_queue"]
    assert max(item["risk_score"] for item in data["trace_analytics"]) >= 5
