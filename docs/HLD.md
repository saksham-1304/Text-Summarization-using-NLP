# High-Level Design (HLD) - Text Summarization System

## 1. System Overview

A production-grade **dialogue summarization system** that fine-tunes `facebook/bart-large-cnn` on the **SAMSum** dataset (16k messenger-like conversations with human-written summaries). The system provides a REST API and web UI for real-time inference.

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │  Web UI  │  │ REST API     │  │  Swagger/OpenAPI Docs     │  │
│  │ (HTML/JS)│  │ (FastAPI)    │  │  (auto-generated)         │  │
│  └────┬─────┘  └──────┬───────┘  └───────────┬───────────────┘  │
│       │               │                      │                   │
│       └───────────────┼──────────────────────┘                   │
│                       │                                          │
│              ┌────────▼────────┐                                 │
│              │   API Gateway   │                                 │
│              │   (Uvicorn)     │                                 │
│              └────────┬────────┘                                 │
│                       │                                          │
│  ┌────────────────────▼─────────────────────────┐                │
│  │           INFERENCE ENGINE                    │                │
│  │  ┌─────────────────────────────────────────┐  │                │
│  │  │  PredictionPipeline                     │  │                │
│  │  │  - BART-large-CNN (fine-tuned)          │  │                │
│  │  - Greedy Decoding (num_beams=1)        │  │                │
│  │  - FP16 + Accelerate Offloading         │  │                │
│  │  └─────────────────────────────────────────┘  │                │
│  └───────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                           │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐  ┌─────┐  │
│  │  Stage 1 │→ │ Stage 2  │→ │ Stage 3  │→ │Stage 4│→ │St. 5│  │
│  │  Data    │  │  Data    │  │  Data    │  │Model  │  │Model│  │
│  │ Ingest   │  │ Validate │  │Transform │  │Train  │  │Eval │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────┘  └─────┘  │
│       │                                         │         │      │
│       ▼                                         ▼         ▼      │
│  ┌──────────┐                            ┌──────────┐ ┌──────┐   │
│  │ HF Hub   │                            │ Model    │ │ROUGE │   │
│  │ (SAMSum) │                            │ Artifacts│ │Scores│   │
│  └──────────┘                            └──────────┘ └──────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Component Descriptions

### 3.1 Data Layer
| Component | Description |
|-----------|-------------|
| **HuggingFace Hub** | Source for SAMSum dataset (samsum) |
| **Local Artifact Store** | `artifacts/` directory for intermediate data |
| **Arrow Format** | Dataset stored in Apache Arrow for fast I/O |

### 3.2 Training Pipeline
| Stage | Input | Output | Description |
|-------|-------|--------|-------------|
| **Data Ingestion** | HF Hub | Raw SAMSum on disk | Downloads and caches dataset |
| **Data Validation** | Raw dataset | Validation report (JSON) | Schema, split, quality checks |
| **Data Transformation** | Raw dataset | Tokenized dataset | BART tokenization (max 1024/128 tokens) |
| **Model Training** | Tokenized data | Fine-tuned model | BART fine-tuning with early stopping |
| **Model Evaluation** | Test split + model | ROUGE metrics (CSV/JSON) | ROUGE-1/2/L/Lsum evaluation |

### 3.3 Inference Layer
| Component | Description |
|-----------|-------------|
| **PredictionPipeline** | Loads model once, serves predictions |
| **FastAPI Server** | Async REST API with OpenAPI docs |
| **Web UI** | HTML/CSS/JS frontend for interactive use |

## 4. Data Flow

```
User Input (dialogue text)
    │
    ▼
FastAPI /predict endpoint
    │
    ▼
PredictionPipeline.predict()
    │
    ├── Tokenize input (BART tokenizer)
    ├── Generate summary (greedy decoding, no_repeat_ngram_size=5)
    └── Decode tokens to text
    │
    ▼
JSON Response (summary + metadata)
```

## 5. Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Language** | Python | 3.10+ |
| **ML Framework** | PyTorch | 2.0+ |
| **NLP Library** | HuggingFace Transformers | 4.36+ |
| **Dataset** | HuggingFace Datasets | 2.16+ |
| **API Framework** | FastAPI | 0.109+ |
| **Server** | Uvicorn | 0.25+ |
| **Containerization** | Docker | Multi-stage |
| **Testing** | Pytest | 7.4+ |
| **CI/CD** | GitHub Actions | Latest |

## 6. Non-Functional Requirements

| Requirement | Target |
|------------|--------|
| **Inference Latency** | < 3s for single dialogue (GPU) |
| **API Throughput** | 10+ req/s on single GPU |
| **Model Size** | ~1.6 GB (BART-large-CNN) |
| **Dataset Size** | ~16k examples (SAMSum) |
| **Availability** | Docker health checks, graceful degradation |

## 7. Deployment Options

1. **Local Development**: `python app.py`
2. **Docker**: `docker build -t summarizer . && docker run -p 8080:8080 summarizer`
3. **Kaggle/Colab**: Use provided notebook for GPU training
4. **Cloud**: Deploy Docker image to AWS ECS, GCP Cloud Run, or Azure Container Apps
