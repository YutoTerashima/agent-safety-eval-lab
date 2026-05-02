from pathlib import Path
import csv
import json
import re


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(
    r"(^|/)(TOKEN|token)(\.|$)|kaggle\.json|(^|/)data/raw(/|$)|"
    r"\.safetensors$|\.ckpt$|optimizer\.pt$|\.arrow$|\.pth$|\.pt$",
    re.IGNORECASE,
)


def test_top_tier_review_gate_artifacts_exist():
    expected = [
        "docs/top_tier_reviewer_packet.md",
        "reports/results/claim_evidence_matrix.csv",
        "reports/results/artifact_manifest.json",
        "reports/results/reproducibility_manifest.json",
        "reports/results/top_tier_quality_gate.json",
        "reports/figures/top_tier_review_scores.png",
    ]
    for relative in expected:
        path = ROOT / relative
        assert path.exists(), relative
        assert path.stat().st_size > 0, relative


def test_claim_evidence_matrix_passes_review_gate():
    with (ROOT / "reports/results/claim_evidence_matrix.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 8
    assert all(row["status"] in {"pass", "qualified-pass"} for row in rows)
    assert all(row["evidence_paths"].strip() for row in rows)


def test_artifact_manifest_is_clean_and_hashable():
    data = json.loads((ROOT / "reports/results/artifact_manifest.json").read_text(encoding="utf-8"))
    assert len(data["artifacts"]) >= 10
    for item in data["artifacts"]:
        assert item["sha256"]
        assert item["size_bytes"] < 100 * 1024 * 1024
        assert not FORBIDDEN.search(item["path"])


def test_reproducibility_manifest_has_highest_standard_gate():
    data = json.loads((ROOT / "reports/results/reproducibility_manifest.json").read_text(encoding="utf-8"))
    assert data["review_threshold"]["highest_standard_gate"] == ">=95/100"
    assert data["review_threshold"]["status"] == "pass"
    assert "pytest" in data["commands"]["test"]
