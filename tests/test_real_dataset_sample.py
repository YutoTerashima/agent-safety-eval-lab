import json
from pathlib import Path


def test_beavertails_feature_sample_has_real_coverage():
    rows = [json.loads(line) for line in Path("datasets/external/beavertails_feature_sample.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 200
    assert any(not row["is_safe"] for row in rows)
    assert any(row["categories"] for row in rows)
