install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

coverage:
	pytest --cov=otto_forecasting --cov-report=term-missing --cov-fail-under=65

quality:
	python scripts/check_quality.py

smoke-data:
	otto-forecast make-smoke-data --output data/processed/synthetic_hourly.csv --hours 500

smoke: smoke-data
	otto-forecast research --config configs/smoke.yaml

aggregate:
	otto-forecast aggregate --input data/raw/otto-recsys-train.jsonl --output data/processed/otto_hourly.csv --frequency 1h

audit:
	otto-forecast audit-data --input data/processed/otto_hourly.csv --output artifacts/data_audit.json

train:
	otto-forecast train --config configs/default.yaml

research:
	otto-forecast research --config configs/research.yaml

ablate:
	otto-forecast ablate --config configs/research.yaml

api:
	OTTO_ARTIFACT_DIR=artifacts/research uvicorn otto_forecasting.api:app --host 0.0.0.0 --port 8000
