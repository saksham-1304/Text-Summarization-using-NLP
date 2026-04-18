# FINAL SUBMISSION GUIDE

This guide is the final, non-redundant checklist for local validation, Kaggle training, and submission.

## 1) What Was Improved

The project now includes stronger anti-overfitting and preprocessing behavior:

- Stage-3 text normalization for dialogue and summary fields.
- Training-only augmentation (no validation/test leakage).
- Augmentation operations:
  - light paraphrasing
  - filler-word removal
  - partial turn shuffling
  - light noise injection
- Seeded augmentation behavior for reproducibility.
- Lightweight stage-4 smoke mode for fast local safety checks.
- API test startup no longer loads full model (stable CI/local tests).

## 2) Final Local Validation (Windows PowerShell)

Run from project root:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python -m pytest tests -q
python scripts/local_preflight.py
```

Expected:

- Tests: `38 passed, 16 skipped`
- Preflight: `PREFLIGHT SUCCESS: local environment is ready for Kaggle training`

## 3) Optional Data Preparation Before Kaggle

If you want to validate data stages locally first:

```powershell
python main.py --stage 1 --to 3
```

## 4) Kaggle Training Steps

Recommended: use the maintained notebook `notebooks/kaggle_training.ipynb`
as the single source of truth for Kaggle execution.

1. Create a Kaggle notebook.
2. Enable GPU: `Settings -> Accelerator -> GPU T4 x2`.
3. Add and run these cells:

```python
# Cell 1: dependencies
!pip install -q -r requirements.txt
```

```python
# Cell 2: clone repository
!git clone https://github.com/your-username/Text-Summarization-NLP-Project.git
%cd Text-Summarization-NLP-Project
!python -m pip install -q --upgrade pip
!pip install -q -r requirements.txt
!pip install -e .
```

```python
# Cell 3: run full pipeline
!python main.py
```

Approximate runtime on T4 GPU:

- Stage 1: 2 min
- Stage 2: 1 min
- Stage 3: 6 to 10 min
- Stage 4: 45 to 70 min
- Stage 5: 5 to 10 min

## 5) What To Check After Training

Confirm these artifacts exist:

- `bart-samsum-final/` (from notebook save flow)
- `bart-samsum-final.zip` (optional single-file download)
- `artifacts/model_trainer/bart-samsum-model/`
- `artifacts/model_trainer/training_diagnostics.json`
- `artifacts/model_evaluation/metrics.csv`

Quality expectations:

- ROUGE-1 around 0.45+
- ROUGE-L around 0.41+
- stable validation loss trend
- no exploding train/eval loss gap

## 6) Submission Bundle

Include:

- source code under `src/`
- `config/config.yaml`
- `params.yaml`
- `requirements.txt`
- `README.md`
- docs under `docs/`
- trained artifacts under `artifacts/`

Exclude:

- `.venv/`
- `__pycache__/`
- `.git/`
- temporary log files

## 7) Fast Troubleshooting

If `local_preflight.py` fails:

- run `python -m pip check`
- verify paths in `config/config.yaml`
- verify writable `artifacts/`

If Kaggle GPU OOM occurs:

- lower `per_device_train_batch_size`
- keep gradient checkpointing enabled
- keep smoke mode only for local validation, not final quality training

## 8) Final Ready Command Set

Run these once before submission:

```powershell
.venv\Scripts\Activate.ps1
python -m pytest tests -q
python scripts/local_preflight.py
```

If both pass, the project is ready for Kaggle run and final submission.
