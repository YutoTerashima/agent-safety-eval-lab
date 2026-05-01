# Agent Safety GPU Benchmark

GPU-backed safety classifier and rule baseline on BeaverTails safety data.

## Dataset

- Source: `PKU-Alignment/BeaverTails`
- Config: `default`
- Split: `30k_train`

## Reproducibility

```powershell
conda run -n Transformers python scripts/download_data.py --smoke
conda run -n Transformers python scripts/preprocess_data.py --max-samples 384
conda run -n Transformers python scripts/run_experiment.py --device cuda --smoke
conda run -n Transformers python scripts/make_report.py
```

## Generated Artifacts

- Result JSON: `results/gpu_classification_metrics.json`
- Result CSV: `results/gpu_training_curve.csv`
- Figure: `figures/gpu_confusion_matrix.png`

## Result Snapshot

```json
{
  "labels": [
    "safe",
    "unsafe"
  ],
  "train_rows": 288,
  "test_rows": 96,
  "tfidf_logreg_accuracy": 0.6354,
  "tfidf_logreg_macro_f1": 0.6089,
  "gpu_mlp_accuracy": 0.6042,
  "gpu_mlp_macro_f1": 0.3766,
  "gpu_training_seconds": 0.221,
  "history": [
    {
      "epoch": 1,
      "loss": 0.683
    },
    {
      "epoch": 2,
      "loss": 0.6122
    }
  ],
  "true_labels": [
    "unsafe",
    "safe",
    "unsafe",
    "unsafe",
    "unsafe",
    "safe",
    "safe",
    "unsafe",
    "safe",
    "unsafe",
    "unsafe",
    "unsafe",
    "safe",
    "safe",
    "safe",
    "unsafe",
    "unsafe",
    "safe",
    "unsafe",
    "safe",
    "safe",
    "safe",
    "unsafe",
    "unsafe",
    "unsafe",
    "safe",
    "unsafe",
    "safe",
    "unsafe",
    "safe",
    "unsafe",
    "unsafe",
    "unsafe",
    "safe",
    "safe",
    "safe",
    "unsafe",
    "safe",
    "unsafe",
    "unsafe",
    "safe",
    "unsafe",
    "safe",
    "unsafe",
    "safe",
    "unsafe",
    "safe",
    "unsafe",
    "unsafe",
    "safe",
    "unsafe",
    "unsafe",
    "unsafe",
    "safe",
    "safe",
    "safe",
    "unsafe",
    "unsafe",
    "unsafe",
    "safe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "safe",
    "unsafe",
    "unsafe",
    "unsafe",
    "safe",
    "safe",
    "unsafe",
    "unsafe",
    "safe",
    "unsafe",
    "unsafe",
    "safe",
    "unsafe",
    "safe",
    "safe",
    "unsafe",
    "safe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "safe",
    "safe",
    "safe",
    "unsafe",
    "unsafe",
    "unsafe"
  ],
  "gpu_pred_labels": [
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
    "unsafe",
```

## Failure Analysis

The experiment stores model disagreements, retrieval misses, or policy-risk examples in the result JSON/CSV files when available. These examples are intentionally kept as previews or structured metadata where the source data can contain unsafe or sensitive text.

## Limitations

- Smoke mode prioritizes reproducibility and runtime over leaderboard-scale performance.
- Raw datasets are downloaded to `data/raw/` and are not committed.
- Metrics should be interpreted as portfolio research baselines, not production claims.
