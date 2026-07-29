# Next Steps Before Dissertation Writing

## What this ZIP contains

The repository is upgraded with the code and documentation needed for a much stronger technical evaluation. It does not contain newly generated real-data results because the raw OTTO data and processed hourly CSV were not included in the uploaded repository.

## Step 1 — Replace your GitHub repository carefully

Create a backup branch before replacing files:

```bash
git checkout -b backup-before-research-upgrade
git add .
git commit -m "Backup original dissertation project"
git checkout main
```

Copy the upgraded files into the repository, review the diff and commit them.

## Step 2 — Put the processed data in place

Expected path:

```text
data/processed/otto_hourly.csv
```

When only the raw JSONL exists, run:

```bash
otto-forecast aggregate \
  --input data/raw/otto-recsys-train.jsonl \
  --output data/processed/otto_hourly.csv
```

## Step 3 — Verify the data and code

```bash
otto-forecast audit-data \
  --input data/processed/otto_hourly.csv \
  --output artifacts/data_audit.json
pytest -q
```

Expected repository test result for this ZIP:

```text
17 passed
```

## Step 4 — Run the full research benchmark

```bash
otto-forecast research --config configs/research.yaml
```

This performs rolling-origin evaluation, five neural seeds, eight model families, statistical comparisons and conformal intervals.

## Step 5 — Run the ablation study

```bash
otto-forecast ablate --config configs/research.yaml
```

This can take longer because it trains multiple Transformer variants.

## Step 6 — Review the evidence before writing

Open:

- `artifacts/research/cv_model_ranking.csv`
- `artifacts/research/research_metrics_summary.csv`
- `artifacts/research/statistical_comparisons.csv`
- `artifacts/research/interval_metrics.csv`
- `artifacts/research/ablations/ablation_metrics_summary.csv`
- `artifacts/research/experiment_manifest.json`

Do not assume the Transformer will win. The dissertation should present the actual selected model.

## Step 7 — Send the completed artifact folder for evaluation

The next technical review should use the generated `artifacts/research/` folder. A revised score above 90 cannot be responsibly assessed until the real experiments complete and the statistical evidence is inspected.
