import json
import sys
from pathlib import Path

from otto_forecasting.cli import main


def test_cli_generates_and_audits_smoke_data(monkeypatch, tmp_path: Path, capsys):
    data_path = tmp_path / "synthetic.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "otto-forecast",
            "make-smoke-data",
            "--output",
            str(data_path),
            "--hours",
            "320",
            "--seed",
            "7",
        ],
    )
    main()
    generated = json.loads(capsys.readouterr().out)
    assert generated["rows"] == 320
    assert data_path.exists()

    audit_path = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "otto-forecast",
            "audit-data",
            "--input",
            str(data_path),
            "--output",
            str(audit_path),
        ],
    )
    main()
    audited = json.loads(capsys.readouterr().out)
    assert audited["rows"] == 320
    assert audit_path.exists()
