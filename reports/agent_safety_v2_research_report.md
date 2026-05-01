# Agent Safety Eval Lab V2 Research Report

## Abstract

This V2 upgrade turns the repository into a reproducible project-level experiment suite. The run records the dataset, device, experiment matrix, metrics, figures, failure analysis, and reproduction commands in committed small artifacts.

## Dataset

- Source path: `data/processed/classification_examples.jsonl`
- Profile: `full`
- Runtime: `24.71` seconds
- Device: `cuda` / `NVIDIA GeForce RTX 5090 Laptop GPU`

## Methods

Experiments declared in `configs/experiment_matrix.yaml`:

- `rule_safety_keywords`: `rule`
- `tfidf_word_lr_prompt_response`: `tfidf_word`
- `tfidf_char_lr_prompt_response`: `tfidf_char`
- `gpu_tfidf_mlp_prompt_response`: `torch_mlp`

## Experiments

The matrix produced `4` result rows. Best observed `macro_f1`: `0.7743` from `tfidf_word_lr_prompt_response`.

## Results

Key artifacts:

- `reports\results\v2_main_results.csv`
- `reports\results\v2_ablation_results.csv`
- `reports\results\v2_failure_cases.json`
- `reports\figures\v2_accuracy_by_experiment.png`
- `reports\figures\v2_confusion_matrix.png`
- `reports\figures\v2_model_macro_f1.png`

## Ablations

Configured ablations: prompt_only, prompt_plus_response, class_weight_balanced, tool_risk_adjusted. The generated ablation files quantify threshold, perturbation, architecture, retrieval, or metric sensitivity depending on the project.

## Failure Analysis

Failure records: `80`.

Top clusters:

- `false_negative`: 67
- `false_positive`: 13

## Discussion

Agent safety needs both local prompt/response classification and trace-level policy interpretation. V2 measures where classifiers miss unsafe responses and how those misses propagate into simulated tool decisions.

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
