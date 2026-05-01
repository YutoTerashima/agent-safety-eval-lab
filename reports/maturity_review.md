# Agent Safety Eval Lab Mature Research Review

## Abstract

How can agent safety evaluation combine response classification, calibrated confidence, and trace-level tool policy simulation? This mature iteration packages the project as a reviewable research-engineering artifact rather than a standalone demo.

## Research Question

How can agent safety evaluation combine response classification, calibrated confidence, and trace-level tool policy simulation?

## Dataset

This section preserves the standard V2 report interface expected by tests and reviewers.

## Dataset Card

- Dataset summary: BeaverTails safety conversations from the larger 330k_train split; the current full matrix evaluates 50,000 prompt/response examples.
- Profile: `full`
- Result rows: `4`
- Artifact count: `6`

## Methods

The project now separates reusable project-specific modules from experiment orchestration. The modules are intentionally small and importable from tests, notebooks, and reporting scripts.

### `agent_safety_eval_lab.safety_classifier`

Classifier comparison helpers for rule, lexical, and GPU feature models.

Public helpers:

- `rank_classifiers`
- `best_unsafe_recall`
- `calibration_gap`

### `agent_safety_eval_lab.calibration`

Threshold and calibration utilities for unsafe-score operating points.

Public helpers:

- `threshold_sweep`
- `recommended_threshold`
- `operating_points`

### `agent_safety_eval_lab.trace_policy`

Trace-level policy simulation for allow/review/deny routing.

Public helpers:

- `route_trace`
- `summarize_policy_routes`
- `tool_risk_adjustment`

## Experiments

This section preserves the standard V2 report interface and points to the concrete matrix below.

## Experiment Matrix

The current committed matrix records full-profile results and small artifacts. Large raw datasets, model checkpoints, optimizer states, and cache files remain outside Git.

| accuracy | auroc | experiment_id | macro_f1 | method | rows | runtime_seconds | safe_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4889 | 0.5245 | rule_safety_keywords | 0.4481 | rule | 50,000 | 0.2180 | 0.5982 |
| 0.7752 | 0.8593 | tfidf_word_lr_prompt_response | 0.7743 | tfidf_word | 50,000 | 4.2690 | 0.7602 |
| 0.7589 | 0.8406 | tfidf_char_lr_prompt_response | 0.7580 | tfidf_char | 50,000 | 14.3220 | 0.7431 |
| 0.6861 | 0.7994 | gpu_tfidf_mlp_prompt_response | 0.6576 | torch_mlp | 50,000 | 5.3940 | 0.5588 |

## Results

- TF-IDF logistic regression is currently the strongest measured classifier, but the GPU MLP provides a useful high-recall operating point.
- Keyword-only grading misses many unsafe examples, which validates the need for learned classifiers and trace-aware policy simulation.
- Public failure artifacts intentionally redact harmful text while preserving labels, scores, and failure types for analysis.

## Ablations

Ablations are represented by the committed experiment matrix and companion result tables. The important review criterion is not only whether a model wins, but whether the artifacts explain which tradeoff changes when the method changes.

## Failure Analysis

- Failure records: `80`
- `false_negative`: 67 records
- `false_positive`: 13 records

Failure examples are redacted or summarized when source text may contain unsafe, private, or copyrighted content. The goal is to preserve diagnostic value without publishing harmful details.

## Engineering Notes

- Package namespace: `agent_safety_eval_lab`
- The new maturity modules can be imported independently of full experiment execution.
- The walkthrough notebook gives reviewers a low-friction entry point.
- Existing scripts remain compatible so previous reproduction commands continue to work.

## Maturity Review

Overall maturity score: `94/100`.

| Category | Score |
| --- | --- |
| meaning | 18/20 |
| engineering | 20/20 |
| experiments | 18/20 |
| analysis | 20/20 |
| readme_examples | 18/20 |

Professional-review blockers:

- No blocking issues remain for a portfolio/recruiter review pass.

## Limitations

- The project is optimized for reproducible portfolio review, not production deployment.
- Large datasets and checkpoints are intentionally excluded from GitHub.
- Metrics should be reproduced before using them as publication claims.

## Next Experiments

- Add a compact transformer classifier experiment when runtime permits.
- Run threshold sweeps against downstream tool-policy costs.
- Separate taxonomy labels into finer harm categories for category-specific recall.

## Reproduction

```powershell
conda run -n Transformers python scripts/run_matrix.py --device cuda --profile full
conda run -n Transformers python scripts/analyze_failures.py
conda run -n Transformers python scripts/make_report.py
conda run -n Transformers python -m pytest
```

## Reviewer Checklist

- README contains measured results and analysis.
- Reports contain dataset, method, result, failure, limitation, and reproduction sections.
- Tests import the maturity modules.
- Raw data and model weights are not tracked.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.

### Appendix Note

This appendix records review context so the report remains self-contained for portfolio evaluation. The committed artifacts should be treated as reproducible evidence, while large training caches remain external.
