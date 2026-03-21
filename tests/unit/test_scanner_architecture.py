from __future__ import annotations

from pathlib import Path


SCANNER_PATH = Path("pages/2_Сканнер_транзакций.py")


def test_scanner_uses_api_batch_and_not_local_model_inference() -> None:
    source = SCANNER_PATH.read_text(encoding="utf-8")

    assert "score_batch_v2" in source
    assert "predict_proba" not in source
    assert "load_artifacts(" not in source
