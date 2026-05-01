import json
from pathlib import Path

from agent_safety_eval_lab.coverage import suite_coverage


if __name__ == "__main__":
    cases = json.loads(Path("datasets/full_trace_suite.json").read_text(encoding="utf-8"))
    result = suite_coverage(cases)
    Path("reports/suite_coverage.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
