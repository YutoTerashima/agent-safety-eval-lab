from __future__ import annotations

"""Project-specific research helpers for Agent Safety Eval Lab.

Purpose: Classifier comparison helpers for rule, lexical, and GPU feature models.
This module is intentionally small and dependency-light so it can be imported
from tests, notebooks, and report-generation scripts without re-running the
full experiment matrix.
"""

from collections import Counter
from typing import Iterable, Mapping, Any


MODULE_PURPOSE = 'Classifier comparison helpers for rule, lexical, and GPU feature models.'
PROJECT_TITLE = 'Agent Safety Eval Lab'


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in row:
            return _as_float(row.get(name))
    return 0.0


def rank_classifiers(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return rows ranked by the strongest available quality signal."""
    materialized = [dict(row) for row in rows]
    for row in materialized:
        row["_quality_score"] = max(
            _metric(row, "macro_f1", "mrr", "completeness_score", "accuracy"),
            _metric(row, "recall@10", "unsafe_recall", "attack_recall"),
        )
    return sorted(materialized, key=lambda row: row["_quality_score"], reverse=True)


def best_unsafe_recall(rows: Iterable[Mapping[str, Any]], label: str = "experiment_id") -> dict[str, float]:
    """Summarize the project-specific tradeoff metrics used in the report."""
    materialized = [dict(row) for row in rows]
    if not materialized:
        return {"rows": 0.0, "best_quality": 0.0, "mean_quality": 0.0}
    ranked = rank_classifiers(materialized)
    scores = [_as_float(row.get("_quality_score")) for row in ranked]
    return {
        "rows": float(len(materialized)),
        "best_quality": max(scores),
        "mean_quality": sum(scores) / len(scores),
        "distinct_labels": float(len({str(row.get(label, "unknown")) for row in materialized})),
    }


def calibration_gap(records: Iterable[Mapping[str, Any]], field: str = "failure_type") -> list[dict[str, Any]]:
    """Create a compact count table for errors, routes, groups, or artifacts."""
    counts = Counter(str(row.get(field, "unknown")) for row in records)
    return [
        {"name": name, "count": count}
        for name, count in counts.most_common()
    ]
