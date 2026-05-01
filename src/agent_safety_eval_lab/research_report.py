from __future__ import annotations

"""Report metadata for the mature portfolio iteration."""

PROJECT_TITLE = 'Agent Safety Eval Lab'
RESEARCH_PROBLEM = 'How can agent safety evaluation combine response classification, calibrated confidence, and trace-level tool policy simulation?'
DATASET_SUMMARY = 'BeaverTails safety conversations from the larger 330k_train split; the current full matrix evaluates 50,000 prompt/response examples.'
TAKEAWAYS = ['TF-IDF logistic regression is currently the strongest measured classifier, but the GPU MLP provides a useful high-recall operating point.', 'Keyword-only grading misses many unsafe examples, which validates the need for learned classifiers and trace-aware policy simulation.', 'Public failure artifacts intentionally redact harmful text while preserving labels, scores, and failure types for analysis.']
NEXT_EXPERIMENTS = ['Add a compact transformer classifier experiment when runtime permits.', 'Run threshold sweeps against downstream tool-policy costs.', 'Separate taxonomy labels into finer harm categories for category-specific recall.']


def report_outline() -> list[str]:
    return [
        "Abstract",
        "Research question",
        "Dataset card",
        "Methods",
        "Experiment matrix",
        "Results",
        "Ablations",
        "Failure analysis",
        "Engineering notes",
        "Limitations",
        "Reproduction",
    ]


def maturity_claims() -> dict[str, object]:
    return {
        "title": PROJECT_TITLE,
        "problem": RESEARCH_PROBLEM,
        "dataset": DATASET_SUMMARY,
        "takeaways": TAKEAWAYS,
        "next_experiments": NEXT_EXPERIMENTS,
    }
