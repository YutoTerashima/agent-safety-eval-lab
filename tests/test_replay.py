from pathlib import Path

from agent_safety_eval_lab.replay import replay_directory


def test_replay_directory_detects_failures():
    results = replay_directory(Path("examples/traces"))
    verdicts = {item["case_id"]: item["verdict"] for item in results}
    assert verdicts["trace-public-summary"] == "pass"
    assert verdicts["trace-egress-attempt"] == "fail"
