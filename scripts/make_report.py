
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reports" / "results"


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main() -> int:
    cfg = json.loads((REPO / "configs" / "experiment_matrix.yaml").read_text(encoding="utf-8"))
    index = read_json(RESULTS / "experiment_index.json", {})
    failures = read_json(RESULTS / "v2_failure_analysis.json", {"total_failure_records": 0, "clusters": []})
    result_rows = index.get("results", [])
    artifacts = index.get("artifacts", [])
    best_line = ""
    if result_rows:
        metric_keys = ["macro_f1", "recall@5", "accuracy", "token_f1", "completeness_score", "mrr"]
        key = next((k for k in metric_keys if any(k in r for r in result_rows)), None)
        if key:
            best = max((r for r in result_rows if key in r), key=lambda r: r[key])
            best_line = f"Best observed `{key}`: `{best.get(key):.4f}` from `{best.get('experiment_id') or best.get('repo') or best.get('policy_decision')}`."
    report = f"""# {cfg['title']} Research Report

## Abstract

This V2 upgrade turns the repository into a reproducible project-level experiment suite. The run records the dataset, device, experiment matrix, metrics, figures, failure analysis, and reproduction commands in committed small artifacts.

## Dataset

- Source path: `{cfg.get('dataset_path')}`
- Profile: `{index.get('profile')}`
- Runtime: `{index.get('total_runtime_seconds')}` seconds
- Device: `{index.get('device', {}).get('actual_device')}` / `{index.get('device', {}).get('gpu_name')}`

## Methods

Experiments declared in `configs/experiment_matrix.yaml`:

""" + "\n".join(f"- `{e.get('id')}`: `{e.get('method') or {k:v for k,v in e.items() if k != 'id'}}`" for e in cfg.get("experiments", [])) + f"""

## Experiments

The matrix produced `{len(result_rows)}` result rows. {best_line}

## Results

Key artifacts:

""" + "\n".join(f"- `{artifact}`" for artifact in artifacts[:20]) + f"""

## Ablations

Configured ablations: {', '.join(cfg.get('ablations', []))}. The generated ablation files quantify threshold, perturbation, architecture, retrieval, or metric sensitivity depending on the project.

## Failure Analysis

Failure records: `{failures.get('total_failure_records', 0)}`.

Top clusters:

""" + "\n".join(f"- `{c.get('cluster')}`: {c.get('count')}" for c in failures.get("clusters", [])[:10]) + f"""

## Discussion

{cfg.get('discussion', '')}

## Limitations

- Full raw caches, model weights, and optimizer states are intentionally excluded from GitHub.
- Results are designed for reproducible portfolio research; they are not production safety, medical, or compliance guarantees.
- Some V2 experiments use compact local artifacts to keep the repository lightweight.

## Reproduction

```powershell
conda run -n Transformers python scripts/run_matrix.py --device cuda --profile full
conda run -n Transformers python scripts/analyze_failures.py
conda run -n Transformers python scripts/make_report.py
conda run -n Transformers python -m pytest
```
"""
    out = REPO / "reports" / cfg.get("report", "v2_research_report.md")
    out.write_text(report, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
