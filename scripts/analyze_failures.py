
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reports" / "results"


def main() -> int:
    candidates = [
        RESULTS / "v2_failure_cases.json",
        RESULTS / "v2_retrieval_failures.json",
        RESULTS / "v2_metric_disagreements.json",
        RESULTS / "v2_trace_failures.json",
        RESULTS / "v2_clustered_failures.json",
        RESULTS / "v2_misclassified_reviews.json",
    ]
    rows = []
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(payload if isinstance(payload, list) else [payload])
    buckets = Counter()
    for row in rows:
        key = row.get("failure_type") or row.get("failure_reason") or row.get("perturbation") or row.get("method") or row.get("experiment_id") or "case"
        buckets[str(key)] += 1
    summary = {
        "total_failure_records": len(rows),
        "clusters": [{"cluster": key, "count": count} for key, count in buckets.most_common()],
        "examples": rows[:25],
    }
    (RESULTS / "v2_failure_analysis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
