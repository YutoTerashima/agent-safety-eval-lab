from agent_safety_eval_lab.report import build_markdown


def test_report_contains_case_table():
    md = build_markdown({"cases": 1, "passed": 1, "failed": 0, "high_risk": 0}, ["C-1: pass | pass"])
    assert "| Case | Status | Risk | Findings |" in md
    assert "C-1" in md
