# Final GitHub and Execution Guide

This is the exact workflow for replacing the existing GitHub repository safely, executing the real OTTO experiments, preserving the evidence, and publishing only the final reproducible outputs.

## Important boundaries

- Do not upload `otto-recsys-train.jsonl`, `otto_hourly.csv`, `kaggle.json`, access tokens, `.pt`, `.joblib`, or `.npy` files to GitHub.
- The project forecasts aggregate hourly carts and orders, not individual products or SKUs.
- Synthetic smoke-test results are not dissertation evidence.
- Do not state that any model is best until the real rolling-origin experiment finishes.

## Part A — Replace the current GitHub repository safely on macOS

### 1. Download and unzip the research-grade ZIP

Place the ZIP in `~/Downloads`, then run:

```bash
cd ~/Downloads
rm -rf otto-upgrade
mkdir otto-upgrade
unzip otto-demand-forecasting-research-grade-final.zip -d otto-upgrade
```

The upgraded files should be located at:

```text
~/Downloads/otto-upgrade/otto-demand-forecasting-research-grade/
```

### 2. Clone the existing GitHub repository

```bash
cd ~/Documents
rm -rf otto-demand-forecasting

git clone https://github.com/nikchey29/otto-demand-forecasting.git
cd otto-demand-forecasting
```

### 3. Create and push a backup branch

```bash
git switch -c backup-before-research-upgrade
git push -u origin backup-before-research-upgrade
git switch main
```

Do not skip this step. The original project will remain recoverable from the backup branch.

### 4. Replace the working files while preserving Git history

```bash
rsync -av --delete \
  --exclude='.git' \
  ~/Downloads/otto-upgrade/otto-demand-forecasting-research-grade/ \
  ./
```

Check the replacement:

```bash
git status
git diff --stat
```

### 5. Verify that no private or large data files are staged

```bash
find . -type f \( \
  -name 'kaggle.json' -o \
  -name '*.pt' -o \
  -name '*.joblib' -o \
  -name '*.npy' -o \
  -name 'otto-recsys-train.jsonl' -o \
  -name 'otto_hourly.csv' \
\) -print
```

The command should not display any files that will be committed.

### 6. Install and test locally

Python 3.11 or newer is required.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
pytest -q
python scripts/check_quality.py
```

Expected test result for the supplied repository:

```text
17 passed
```

### 7. Commit and push the upgraded project

```bash
git add -A
git status
git commit -m 'Upgrade dissertation forecasting project to research-grade evaluation'
git push origin main
```

### 8. Verify GitHub

On the GitHub repository page, check that:

- `README.md` displays correctly.
- `src/otto_forecasting/` exists.
- `configs/research.yaml` exists.
- `tests/` exists.
- `notebooks/OTTO_Dissertation_Final_Run_Colab.ipynb` exists.
- The Actions tab shows the CI workflow passing.
- The backup branch `backup-before-research-upgrade` exists.

## Part B — Prepare persistent Google Drive storage

Create this folder structure in Google Drive:

```text
My Drive/
└── OTTO_Dissertation/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── artifacts/
    └── logs/
```

The preferred reusable input is:

```text
My Drive/OTTO_Dissertation/data/processed/otto_hourly.csv
```

The raw file, when needed, is:

```text
My Drive/OTTO_Dissertation/data/raw/otto-recsys-train.jsonl
```

## Part C — Execute in Google Colab

Open:

```text
notebooks/OTTO_Dissertation_Final_Run_Colab.ipynb
```

Use **Open in Colab**, then execute the cells from top to bottom.

### 1. Select a GPU runtime

Use a GPU runtime for GRU and Transformer training. Aggregation of JSONL is CPU-bound.

### 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Clone the final repository and install it

```python
%cd /content
!rm -rf otto-demand-forecasting
!git clone https://github.com/nikchey29/otto-demand-forecasting.git
%cd /content/otto-demand-forecasting
!python -m pip install --upgrade pip
!pip install -e '.[dev]'
```

### 4. Confirm the GPU

```python
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
```

### 5. Copy the processed data to Colab local storage

This is the preferred route because repeatedly reading the raw JSONL is unnecessary.

```python
!mkdir -p data/processed
!cp '/content/drive/MyDrive/OTTO_Dissertation/data/processed/otto_hourly.csv' \
     data/processed/otto_hourly.csv
!ls -lh data/processed/otto_hourly.csv
```

### 6. Only when the processed CSV does not exist: aggregate the raw JSONL

Copy the raw file from Drive:

```python
!mkdir -p data/raw data/processed
!cp '/content/drive/MyDrive/OTTO_Dissertation/data/raw/otto-recsys-train.jsonl' \
     data/raw/otto-recsys-train.jsonl
```

Aggregate it:

```python
!otto-forecast aggregate \
  --input data/raw/otto-recsys-train.jsonl \
  --output data/processed/otto_hourly.csv \
  --frequency 1h
```

Immediately preserve the processed CSV:

```python
!cp data/processed/otto_hourly.csv \
  '/content/drive/MyDrive/OTTO_Dissertation/data/processed/otto_hourly.csv'
```

### 7. Audit the real data

```python
!mkdir -p artifacts
!otto-forecast audit-data \
  --input data/processed/otto_hourly.csv \
  --output artifacts/data_audit.json
```

### 8. Run tests and the synthetic smoke test

```python
!pytest -q
!python scripts/check_quality.py
!otto-forecast make-smoke-data \
  --output data/processed/synthetic_hourly.csv \
  --hours 500
!otto-forecast research --config configs/smoke.yaml
```

The smoke run confirms code execution only. Do not use its metrics in the dissertation.

### 9. Create a Colab configuration that writes results to Drive

```python
from pathlib import Path
import yaml

source = Path('configs/research.yaml')
config = yaml.safe_load(source.read_text())
config['data']['processed_path'] = 'data/processed/otto_hourly.csv'
config['output_dir'] = '/content/drive/MyDrive/OTTO_Dissertation/artifacts'
Path('configs/research_colab.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
print(Path('configs/research_colab.yaml').read_text())
```

### 10. Run the full research benchmark

```python
!otto-forecast research --config configs/research_colab.yaml \
  2>&1 | tee '/content/drive/MyDrive/OTTO_Dissertation/logs/research_run.log'
```

Do not change the model configuration after inspecting the final holdout unless the change is documented and a new untouched holdout is available.

### 11. Run the ablation study

```python
!otto-forecast ablate --config configs/research_colab.yaml \
  2>&1 | tee '/content/drive/MyDrive/OTTO_Dissertation/logs/ablation_run.log'
```

### 12. Confirm the final evidence

```python
!find '/content/drive/MyDrive/OTTO_Dissertation/artifacts/research' \
  -maxdepth 2 -type f -printf '%p\n' | sort
```

Required outputs include:

```text
cv_model_ranking.csv
research_metrics_raw.csv
research_metrics_summary.csv
research_predictions.csv
statistical_comparisons.csv
interval_metrics.csv
selected_model_horizon_metrics.csv
run_metadata.csv
experiment_manifest.json
model_comparison_repeated.png
cv_fold_stability.png
selected_model_horizon_error.png
ablations/ablation_metrics_raw.csv
ablations/ablation_metrics_summary.csv
```

## Part D — Publish final experimental evidence to GitHub

Do this from the Mac clone, not by exposing a GitHub token inside Colab.

### 1. Download the final research folder from Google Drive

Download:

```text
My Drive/OTTO_Dissertation/artifacts/research
```

Place it temporarily at:

```text
~/Downloads/research
```

### 2. Copy publishable evidence into the cloned repository

```bash
cd ~/Documents/otto-demand-forecasting
rsync -av --delete \
  --exclude='*.pt' \
  --exclude='*.joblib' \
  --exclude='*.npy' \
  ~/Downloads/research/ \
  artifacts/research/
```

The `.gitignore` is designed to allow research CSV, JSON and PNG evidence while excluding binary models and arrays.

### 3. Verify the final evidence before committing

```bash
git status
find artifacts/research -type f -maxdepth 2 -print | sort
```

Confirm that no `.pt`, `.joblib`, `.npy`, raw JSONL, processed CSV, token or credential file is staged.

### 4. Commit and push the real results

```bash
git add artifacts/research
git commit -m 'Add final rolling-origin evaluation and ablation results'
git push origin main
```

### 5. Verify GitHub Actions again

Wait for the CI workflow to pass after the results commit.

## Part E — Final repository checklist

The repository is technically final only when all boxes below are satisfied:

- [ ] Main branch contains the research-grade source code.
- [ ] Backup branch contains the original project.
- [ ] CI is passing.
- [ ] `17 passed` is reproducible.
- [ ] Real-data `data_audit.json` exists.
- [ ] Rolling-origin results exist.
- [ ] Five neural seeds completed.
- [ ] Statistical comparison intervals exist.
- [ ] Conformal interval results exist.
- [ ] Ablation results exist.
- [ ] README claims match the generated CSV files.
- [ ] Raw data and credentials are absent from GitHub.
- [ ] AI assistance is disclosed according to university rules.
- [ ] Every design choice can be explained during the viva.

A 90+ mark cannot be guaranteed by repository structure alone. The real results, critical interpretation, dissertation writing, referencing and viva performance determine the final assessment.
