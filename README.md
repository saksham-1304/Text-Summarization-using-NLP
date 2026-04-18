# 📝 Text Summarization NLP Project

> Production-grade **dialogue summarization** system using **BART-large-CNN** fine-tuned on the **SAMSum** dataset. Features a complete ML pipeline with data ingestion, validation, transformation, training, evaluation, REST API, and modern web UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training Pipeline](#training-pipeline)
- [API Documentation](#api-documentation)
- [Web UI](#web-ui)
- [Training on Kaggle/Colab](#training-on-kagglecolab)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [Model Performance](#model-performance)
- [System Design](#system-design)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a complete **end-to-end NLP pipeline** for dialogue summarization:

| What | Details |
|------|---------|
| **Model** | `facebook/bart-large-cnn` (406M params) |
| **Dataset** | [SAMSum](https://huggingface.co/datasets/knkarthick/samsum) — 16,369 messenger-like conversations with human-written summaries |
| **Task** | Given a dialogue between 2+ people, generate a concise 1-2 sentence summary |
| **API** | FastAPI with OpenAPI docs, single + batch endpoints |
| **Web UI** | Modern dark-theme interface for interactive summarization |

### Example

**Input Dialogue:**
```
Amanda: Hey, are we meeting today?
Jerry: Sure! What time works for you?
Amanda: How about 3pm at the coffee shop?
Jerry: Perfect, see you there!
Amanda: Great, I'll bring the project reports.
```

**Generated Summary:**
> Amanda and Jerry will meet at 3pm at the coffee shop. Amanda will bring the project reports.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                       │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────┐  ┌───┐ │
│  │ Stage 1  │→ │ Stage 2  │→ │ Stage 3  │→ │  4  │→ │ 5 │ │
│  │ Ingest   │  │ Validate │  │Transform │  │Train│  │Eval│ │
│  │ (HF Hub) │  │ (Schema) │  │(Tokenize)│  │(GPU)│  │ROUGE│ │
│  └──────────┘  └──────────┘  └──────────┘  └─────┘  └───┘ │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                     INFERENCE API                           │
│                                                             │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │  Web UI  │  │  REST API    │  │  OpenAPI Docs       │   │
│  │ (HTML/JS)│  │  (FastAPI)   │  │  (auto-generated)   │   │
│  └──────────┘  └──────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

- **5-Stage ML Pipeline** with checkpoints, logging, and validation at each stage
- **Config-driven** architecture — change model/dataset/hyperparameters via YAML files
- **REST API** with Pydantic models, error handling, CORS, OpenAPI docs
- **Web UI** with modern dark theme, example loading, keyboard shortcuts
- **Data Validation** — schema checks, null detection, statistics logging
- **Configurable Preprocessing** — text normalization + training-only augmentation in stage 3
- **Early Stopping** — prevents overfitting with patience-based stopping
- **FP16 Training** — 2x speedup on compatible GPUs
- **Gradient Checkpointing** — reduced memory usage for large models
- **Multi-stage Docker** — optimized container builds
- **GitHub Actions CI** — automated testing on Python 3.10/3.11/3.12
- **Comprehensive Tests** — entity, utils, config, and API endpoint tests
- **System Design Docs** — HLD, LLD, and System Design documents
- **Stage-wise Execution** — run specific stages with `--stage` and `--to` flags

---

## 📊 Model Performance & Research Documentation

### Results Summary

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **ROUGE-1** | 0.451 | 0.45+ | ✅ PASS |
| **ROUGE-2** | 0.209 | 0.21+ | ⚠️ MARGINAL |
| **ROUGE-L** | 0.412 | 0.41+ | ✅ PASS |
| **Inference Speed** | 45ms | <100ms | ✅ PASS |

Validated locally with:
- Full test suite: `python -m pytest tests -q`
- Stage 4 smoke training: `python main.py --stage 4 --to 4 --smoke-train`
- Optional full preflight wrapper: `python scripts/local_preflight.py`

### Overfitting Control Techniques

- Early stopping with configurable patience and threshold
- Label smoothing (`label_smoothing_factor`) to reduce over-confident generation
- Gradient clipping (`max_grad_norm`) for stable optimization
- Validation-first checkpoint selection (`load_best_model_at_end` + `eval_loss`)
- Training-only augmentation in stage 3 (paraphrasing, filler removal, turn shuffling, light noise injection)
- Decoding sweep on validation split before final test scoring (stage 5)
- Reproducible seed-controlled training

### 📚 Research Documentation

**READ THESE for interview/viva preparation:**

1. **[Hyperparameter Justification & Ablation Study](docs/HYPERPARAMETER_JUSTIFICATION.md)**
   - Complete ablation studies for epochs, learning rate, batch size
   - Comparison to published BART results
   - Justification for every hyperparameter with evidence
   - ~5 min read

2. **[Results & Evaluation Analysis](docs/RESULTS_AND_EVALUATION.md)**
   - Comprehensive ROUGE metrics breakdown
   - Error analysis with example failure cases
   - Per-category performance (dialogue type, length)
   - Recommendations for future improvements
   - ~7 min read

3. **[System Design Documentation](docs/SYSTEM_DESIGN.md)**
   - Architecture overview
   - Component interactions
   - Data flow diagrams
   - Scalability considerations

4. **[High-Level Design (HLD)](docs/HLD.md)**
   - Business requirements
   - Solution architecture
   - Technology choices

5. **[Low-Level Design (LLD)](docs/LLD.md)**
   - Component implementation details
   - Algorithm specifications
   - Database schema (if applicable)

6. **[Kaggle Prelaunch Checklist](docs/KAGGLE_PRELAUNCH_CHECKLIST.md)**
   - Local preflight commands
   - Overfitting guardrails
   - Kaggle runtime safety checks

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML Framework | PyTorch 2.0+ |
| NLP | HuggingFace Transformers 4.36+ |
| Dataset | HuggingFace Datasets (SAMSum) |
| Evaluation | `evaluate` library (ROUGE metrics) |
| API | FastAPI 0.109+ |
| Server | Uvicorn (ASGI) |
| Frontend | HTML5, CSS3, Vanilla JS |
| Container | Docker (multi-stage) |
| CI/CD | GitHub Actions |
| Testing | Pytest + Coverage |
| Config | YAML + python-box (dot-access) |

---

## Project Structure

```
Text-Summarization-NLP-Project/
├── app.py                          # FastAPI server (API + Web UI)
├── main.py                         # Training pipeline CLI
├── config/
│   └── config.yaml                 # Infrastructure config (paths, model names)
├── params.yaml                     # Hyperparameters (epochs, lr, batch size)
├── src/
│   └── textSummarizer/
│       ├── constants/              # Path constants
│       ├── entity/                 # Frozen dataclass configs
│       ├── config/                 # ConfigurationManager
│       ├── components/             # Stage implementations
│       │   ├── data_ingestion.py   # Download from HF Hub
│       │   ├── data_validation.py  # Schema + quality validation
│       │   ├── data_transformation.py  # Tokenization
│       │   ├── model_trainer.py    # BART fine-tuning
│       │   └── model_evaluation.py # ROUGE evaluation
│       ├── pipeline/               # Stage orchestrators
│       │   ├── stage_01..05        # Training stages
│       │   └── prediction.py       # Inference pipeline
│       ├── utils/                  # YAML, JSON, directory utils
│       └── logging/                # Rotating file logger
├── templates/                      # HTML templates
├── static/                         # CSS styles
├── tests/                          # Pytest test suite
├── docs/                           # HLD, LLD, System Design
├── .github/workflows/              # CI pipeline
├── Dockerfile                      # Multi-stage Docker
├── requirements.txt                # Dependencies
└── setup.py                        # Package config
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- (Optional) NVIDIA GPU with CUDA for training

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Text-Summarization-NLP-Project.git
cd Text-Summarization-NLP-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Training Pipeline

### Run All Stages

```bash
python main.py
```

### Run Specific Stages

```bash
# Run only data ingestion
python main.py --stage 1 --to 1

# Run stages 1 through 3 (data preparation)
python main.py --stage 1 --to 3

# Run only training and evaluation
python main.py --stage 4 --to 5

# Run fast local smoke training (pre-Kaggle sanity check)
python main.py --stage 4 --to 4 --smoke-train
```

### Local Kaggle Preflight (Recommended)

Before launching a long Kaggle run, validate locally:

```bash
# environment checks + stage-4 smoke training
python scripts/local_preflight.py

# optional: also prepare data artifacts locally first
python scripts/local_preflight.py --prepare-data
```

This catches common breakpoints early:
- dependency/import mismatch
- path/permissions errors under `artifacts/`
- trainer argument regressions
- checkpoint loading issues

Smoke mode is intentionally lightweight for local reliability checks:
- uses a tiny public checkpoint (`hf-internal-testing/tiny-random-bart`) when available
- trains on a tiny synthetic subset with short sequence lengths
- validates the stage-4 training loop without requiring long CPU runs

### Stage 3 Preprocessing Controls

Stage 3 now supports configurable normalization and augmentation from `params.yaml`:

```yaml
DataTransformation:
   max_input_length: 1024
   max_target_length: 128
   text_column: dialogue
   summary_column: summary
   enable_augmentation: true
   augmentation_probability: 0.25
   enable_text_normalization: true
```

Notes:
- Augmentation is applied to the training split only.
- Validation/test splits are normalized but not augmented.
- This prevents data leakage while improving generalization.

### Pipeline Stages

| Stage | Name | Duration | Output |
|-------|------|----------|--------|
| 1 | **Data Ingestion** | ~1 min | `artifacts/data_ingestion/samsum_dataset/` |
| 2 | **Data Validation** | ~10 sec | `artifacts/data_validation/status.txt` |
| 3 | **Data Transformation** | ~5 min | `artifacts/data_transformation/samsum_dataset/` |
| 4 | **Model Training** | ~2 hrs (GPU) | `artifacts/model_trainer/bart-samsum-model/` |
| 5 | **Model Evaluation** | ~10 min | `artifacts/model_evaluation/metrics.csv` |

---

## API Documentation

### Start the API Server

```bash
python app.py
```

The server starts at `http://localhost:8080`. Interactive docs at `http://localhost:8080/docs`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check |
| `GET` | `/info` | Model info |
| `GET` | `/docs` | OpenAPI docs |
| `POST` | `/predict` | Single summarization |
| `POST` | `/predict/batch` | Batch summarization |
| `GET` | `/train` | Trigger training |

### Example API Call

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Amanda: Hey, are we meeting today?\nJerry: Sure! 3pm at the coffee shop.\nAmanda: Perfect, see you there!"}'
```

---

## Web UI

Open `http://localhost:8080` in your browser for the interactive web interface:

- Modern charcoal and orange theme
- Example dialogue loading
- Character count and compression ratio
- Keyboard shortcut: `Ctrl+Enter` to submit

---

## Inference Optimizations

To support running the 1.55GB BART-large-CNN model on machines with limited RAM (e.g., < 2GB free), the inference pipeline (`PredictionPipeline`) implements:
- **Accelerate Device Map**: `device_map="auto"` automatically distributes weights across GPU/CPU/Disk.
- **Disk Offloading**: `offload_folder` swaps weights to disk when RAM is full.
- **FP16 Precision**: `torch.float16` halves the memory footprint.
- **Greedy Decoding**: `num_beams=1` and `no_repeat_ngram_size=5` prevent hallucinations while keeping memory usage low.

---

## Training on Kaggle/Colab

Since BART-large-CNN is a large model, training on CPU is very slow. Use free GPU resources.

For the most reliable end-to-end Kaggle run, use
`notebooks/kaggle_training.ipynb` from this repository.

### Kaggle (Recommended — Free T4 GPU)

1. Create a new Kaggle notebook
2. Enable GPU: **Settings → Accelerator → GPU T4 x2**
3. Run the following cells:

```python
# Cell 1: Clone repository
!git clone https://github.com/your-username/Text-Summarization-NLP-Project.git
%cd Text-Summarization-NLP-Project

# Cell 2: Install dependencies from project pins
!python -m pip install -q --upgrade pip
!pip install -q -r requirements.txt
!pip install -e .

# Cell 3: Run full training pipeline
!python main.py

# Cell 4: (Optional) Run only specific stages
!python main.py --stage 4 --to 5  # Just train + eval
```

### Google Colab (Free T4 GPU)

1. Open a new Colab notebook
2. **Runtime → Change runtime type → T4 GPU**
3. Follow the same cells as Kaggle above

### Training Tips

- **Epoch 1** is the most impactful — most learning happens here
- **3 epochs** is optimal for SAMSum (it's a small dataset)
- **Monitor eval_loss** — should decrease steadily
- **Early stopping** halts training if no improvement for 3 evaluations
- After training, download `artifacts/model_trainer/` for deployment

---

## Docker Deployment

```bash
# Build the image
docker build -t text-summarizer .

# Run on CPU
docker run -p 8080:8080 text-summarizer

# Run with GPU (requires nvidia-docker)
docker run --gpus all -p 8080:8080 text-summarizer
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src/textSummarizer --cov-report=term-missing

# Run specific test files
pytest tests/test_entity.py -v
pytest tests/test_utils.py -v
pytest tests/test_config.py -v
pytest tests/test_api.py -v
```

---

## Model Performance

### Current Validated Scores (This Repository)

| Metric | Score | Description |
|--------|-------|-------------|
| **ROUGE-1** | 0.451 | Unigram overlap |
| **ROUGE-2** | 0.209 | Bigram overlap |
| **ROUGE-L** | 0.412 | Longest common subsequence |
| **ROUGE-Lsum** | 0.407 | Sentence-level LCS |

Reference ranges for strong BART runs on SAMSum are discussed in
[docs/HYPERPARAMETER_JUSTIFICATION.md](docs/HYPERPARAMETER_JUSTIFICATION.md)
and [docs/RESULTS_AND_EVALUATION.md](docs/RESULTS_AND_EVALUATION.md).

---

## System Design

Detailed design documents in `docs/`:

| Document | Description |
|----------|-------------|
| [HLD.md](docs/HLD.md) | High-Level Design — architecture, components, data flow |
| [LLD.md](docs/LLD.md) | Low-Level Design — class diagrams, sequence diagrams, patterns |
| [SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) | Full system design — requirements, scaling, security, cost |

---

## Configuration

### config.yaml — Infrastructure Configuration

- Dataset name and storage paths
- Model checkpoint names
- Output directories for each stage

### params.yaml — Training Hyperparameters

- Epochs, batch size, learning rate
- Gradient accumulation, warmup steps
- Tokenization max lengths
- Evaluation batch size and columns

Both files use dot-access via `python-box` for clean code.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.

---

## Acknowledgments

- [SAMSum Dataset](https://huggingface.co/datasets/samsum) — Samsung R&D Institute Poland
- [BART Paper](https://arxiv.org/abs/1910.13461) — Facebook AI Research
- [HuggingFace Transformers](https://huggingface.co/transformers)
- [FastAPI](https://fastapi.tiangolo.com)
