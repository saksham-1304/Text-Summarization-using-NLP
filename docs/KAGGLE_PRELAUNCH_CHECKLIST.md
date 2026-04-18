# Kaggle Prelaunch Checklist

## Objective

Use this checklist to reduce runtime failures and avoid overfitting before starting a long Kaggle training job.

## 1. Local Verification Commands

Run these from the project root.

```bash
python -m pytest tests -q
python scripts/local_preflight.py
python scripts/submission_validator.py
```

Optional full data-prep validation:

```bash
python scripts/local_preflight.py --prepare-data
```

## 2. Minimum Acceptance Criteria

- Tests pass: no failing test in `tests/`
- Local preflight passes
- Submission validator passes (no failed checks)
- Stage 4 smoke training passes
- `artifacts/model_trainer/training_diagnostics.json` is generated

Notebook consistency checks:
- Use `notebooks/kaggle_training.ipynb` for Kaggle training.
- Set `REPO_URL` and `REPO_DIR` in the notebook to your actual fork/repository.

Note: smoke training is a fast stability check (tiny model + tiny synthetic data),
not a quality benchmark. Final model quality should be judged from full Kaggle
training on SAMSum.

## 3. Overfitting Guardrails in This Codebase

These controls are configurable in `params.yaml` under `TrainingArguments`.

- `early_stopping_patience`
- `early_stopping_threshold`
- `label_smoothing_factor`
- `max_grad_norm`
- `weight_decay`
- `load_best_model_at_end`
- `metric_for_best_model = eval_loss` (enforced in trainer)

## 4. Pre-Kaggle Config Sanity

Recommended defaults for stable training:

- `num_train_epochs: 3`
- `learning_rate: 2e-5`
- `gradient_accumulation_steps: 8`
- `label_smoothing_factor: 0.1`
- `max_grad_norm: 1.0`
- `save_total_limit: 3`

## 5. Kaggle Runtime Setup

- Accelerator: GPU T4 x2 (or T4 single GPU)
- Ensure enough free disk before checkpoints
- Keep `save_total_limit` low to avoid disk exhaustion
- If OOM occurs, reduce `per_device_train_batch_size` to 1

## 6. Post-Training Validation

After Kaggle training completes:

- Run stage 5 evaluation (or notebook evaluation cell)
- Confirm metrics file exists at `artifacts/model_evaluation/metrics.csv`
- Confirm decoding sweep artifacts exist:
  - `artifacts/model_evaluation/decoding_sweep_results.csv`
  - `artifacts/model_evaluation/best_decoding_config.json`
- Compare ROUGE against your previous local baseline

## 7. Fast Troubleshooting

- Import errors:
  - reinstall with `pip install -r requirements.txt`
- Model load failures:
  - verify checkpoint path in `config/config.yaml`
- Training instability:
  - lower learning rate to `1e-5`
  - reduce `num_train_epochs`
  - keep label smoothing enabled
