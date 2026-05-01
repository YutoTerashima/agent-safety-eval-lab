import json
from pathlib import Path

from agent_safety_eval_lab.coverage import suite_coverage


def test_full_suite_has_required_coverage():
    cases = json.loads(Path("datasets/full_trace_suite.json").read_text(encoding="utf-8"))
    assert suite_coverage(cases)["coverage_passed"]
