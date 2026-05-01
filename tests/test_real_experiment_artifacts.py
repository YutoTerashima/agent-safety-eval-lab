from pathlib import Path
import json


def test_gpu_experiment_artifacts_exist():
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "reports"
    result_dir = report_dir / "results"
    figure_dir = report_dir / "figures"
    assert result_dir.exists(), "reports/results must exist after the GPU experiment"
    assert any(result_dir.glob("*.json")), "at least one JSON metric artifact is required"
    assert any(result_dir.glob("*.csv")) or (root.name == "open-model-benchmark-cards"), "at least one CSV metric artifact is required"
    assert any(figure_dir.glob("*.png")), "at least one figure is required"
    reports = list(report_dir.glob("*gpu*.md")) + list(report_dir.glob("*real*.md")) + list(report_dir.glob("*benchmark*.md"))
    assert reports, "a markdown experiment report is required"


def test_gpu_metadata_is_recorded():
    root = Path(__file__).resolve().parents[1]
    json_files = list((root / "reports" / "results").glob("*.json"))
    merged = "\n".join(path.read_text(encoding="utf-8") for path in json_files)
    assert "cuda" in merged.lower() or "gpu" in merged.lower() or root.name == "open-model-benchmark-cards"
