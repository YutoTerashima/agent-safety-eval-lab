import json
from collections import Counter
from pathlib import Path

rows = [json.loads(line) for line in Path("datasets/external/beavertails_feature_sample.jsonl").read_text(encoding="utf-8").splitlines()]
categories = Counter(cat for row in rows for cat in row["categories"])
unsafe = sum(not row["is_safe"] for row in rows)
print({"rows": len(rows), "unsafe": unsafe, "top_categories": categories.most_common(5)})
