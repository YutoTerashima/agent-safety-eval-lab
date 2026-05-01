
from __future__ import annotations
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_v2_config_and_index_exist():
    config = REPO / "configs" / "experiment_matrix.yaml"
    index_path = REPO / "reports" / "results" / "experiment_index.json"
    assert config.exists()
    cfg = json.loads(config.read_text(encoding="utf-8"))
    assert len(cfg.get("experiments", [])) >= 3
    assert index_path.exists()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(index.get("results", [])) >= 3
    assert index.get("artifacts")


def test_v2_report_and_figures_exist():
    cfg = json.loads((REPO / "configs" / "experiment_matrix.yaml").read_text(encoding="utf-8"))
    report = REPO / "reports" / cfg["report"]
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    for heading in ["Abstract", "Dataset", "Methods", "Experiments", "Results", "Ablations", "Failure Analysis", "Limitations", "Reproduction"]:
        assert f"## {heading}" in text
    figures = list((REPO / "reports" / "figures").glob("v2_*.png"))
    assert len(figures) >= 3


def test_no_large_v2_artifacts_tracked():
    tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).splitlines()
    banned = (".safetensors", ".arrow", "optimizer.pt", "rng_state.pth", "TOKEN.txt", "kaggle.json")
    assert not [path for path in tracked if path.endswith(banned)]
