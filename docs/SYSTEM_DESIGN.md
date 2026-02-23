# System Design Document - Text Summarization System

## 1. Problem Statement

Build a production-grade **dialogue summarization system** that can:
- Summarize messenger-like conversations into concise 1-2 sentence summaries
- Serve predictions via REST API with < 3 second latency
- Train end-to-end from raw data to deployed model
- Maintain high ROUGE scores (target: ROUGE-1 > 0.45, ROUGE-2 > 0.21)

## 2. Requirements

### 2.1 Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Download and process SAMSum dataset from HuggingFace | Must |
| FR-2 | Validate dataset integrity before training | Must |
| FR-3 | Tokenize dialogues using BART tokenizer | Must |
| FR-4 | Fine-tune BART-large-CNN with configurable hyperparameters | Must |
| FR-5 | Evaluate model using ROUGE metrics | Must |
| FR-6 | Serve predictions via REST API | Must |
| FR-7 | Provide web UI for interactive summarization | Should |
| FR-8 | Support batch prediction | Should |
| FR-9 | Support stage-wise pipeline execution | Should |
| FR-10 | Save checkpoint metrics for comparison | Could |

### 2.2 Non-Functional Requirements
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Inference latency | < 3s per dialogue (GPU) |
| NFR-2 | API availability | 99.9% uptime with health checks |
| NFR-3 | Containerization | Docker with multi-stage build |
| NFR-4 | Testability | >80% code coverage |
| NFR-5 | Reproducibility | Config-driven training, fixed seeds |
| NFR-6 | Scalability | Horizontal via multiple API instances |

## 3. Model Selection & Justification

### 3.1 Why BART-large-CNN?

| Criterion | BART-large-CNN | Pegasus-CNN | T5-base | LED |
|-----------|---------------|-------------|---------|-----|
| **Pre-training** | Denoising autoencoder | Gap sentence gen | Span corruption | Local attention |
| **Base perf** on SAMSum | **ROUGE-1: ~53** | ROUGE-1: ~47 | ROUGE-1: ~45 | ROUGE-1: ~42 |
| **Size** | 406M params | 568M params | 220M params | 162M params |
| **Inference Speed** | Fast | Slow | Fast | Medium |
| **HF Community** | Excellent | Good | Excellent | Limited |

**Decision**: BART-large-CNN offers the best balance of performance, speed, and community support for dialogue summarization.

### 3.2 Why SAMSum Dataset?

| Feature | SAMSum | DialogSum | CNN/DailyMail |
|---------|--------|-----------|---------------|
| **Domain** | Messenger dialogues | Dialogues | News articles |
| **Size** | 16,369 | 13,460 | 311,971 |
| **Avg dialogue** | 124 words | 158 words | 781 words |
| **Avg summary** | 23 words | 26 words | 56 words |
| **Human-written** | Yes | Yes | Highlights |
| **Task match** | Perfect | Good | Poor |

**Decision**: SAMSum is the gold standard for messenger dialogue summarization.

## 4. Training Strategy

### 4.1 Hyperparameters

```yaml
# Optimized for SAMSum + BART-large-CNN
epochs: 3                        # SAMSum is small, 3 epochs avoids overfitting
batch_size: 2                    # Per device (effective: 2 * 8 = 16 with grad accum)
gradient_accumulation: 8         # Memory-efficient large effective batch
learning_rate: 2e-5              # Standard for fine-tuning large models
warmup_steps: 500                # ~3% of total steps
fp16: true                       # 2x speedup on GPU
early_stopping_patience: 3       # Stop if eval loss doesn't improve for 3 evals
max_input_length: 1024           # BART context window
max_target_length: 128           # Summaries are short
```

### 4.2 Training Timeline (Estimated)

| Hardware | Epochs | Time | Notes |
|----------|--------|------|-------|
| **Kaggle T4 GPU** | 3 | ~2 hours | Free tier, recommended |
| **Colab A100** | 3 | ~30 min | Colab Pro |
| **CPU** | 1 | ~24+ hours | Not recommended |
| **RTX 3090** | 3 | ~45 min | Local GPU |

## 5. API Design

### 5.1 Endpoints

```
GET  /                          → Web UI (HTML)
GET  /health                    → {"status": "healthy", "model_loaded": true}
GET  /info                      → Model metadata
GET  /docs                      → OpenAPI documentation
POST /predict                   → Single text summarization
POST /predict/batch             → Batch summarization (up to 10)
GET  /train                     → Trigger training pipeline
```

### 5.2 API Contract

**POST /predict**
```json
// Request
{
    "text": "Amanda: Hey, are we meeting today?\nJerry: Sure! 3pm?"
}

// Response
{
    "input_text": "Amanda: Hey, are we meeting today?\nJerry: Sure! 3pm?",
    "summary": "Amanda and Jerry will meet at 3pm today.",
    "input_length": 55,
    "summary_length": 42,
    "timestamp": "2026-02-24T10:30:00"
}
```

## 6. Scaling Strategy

### 6.1 Vertical Scaling
- Use GPU for inference (10x faster than CPU)
- Enable fp16 inference for 2x memory reduction
- Use ONNX Runtime for 30-50% inference speedup

### 6.2 Horizontal Scaling
```
                    ┌──────────────┐
                    │ Load Balancer│
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼─────┐ ┌───▼─────┐
        │ API Pod 1 │ │API Pod 2│ │API Pod 3│
        │ (GPU)     │ │(GPU)    │ │(GPU)    │
        └───────────┘ └─────────┘ └─────────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼───────┐
                    │ Shared Model │
                    │ (NFS/S3)     │
                    └──────────────┘
```

## 7. Monitoring & Observability

| Aspect | Tool | What to Monitor |
|--------|------|-----------------|
| **Logging** | Python logging (rotating) | Errors, predictions, latency |
| **Health** | /health endpoint | Model loaded, API alive |
| **Metrics** | ROUGE scores (saved) | Model quality over time |
| **Docker** | HEALTHCHECK | Container health |

## 8. Security Considerations

| Threat | Mitigation |
|--------|------------|
| **Input injection** | Pydantic validation, max input length |
| **Resource exhaustion** | Batch size limit (10), input length cap |
| **Model theft** | Don't expose model weights via API |
| **Prompt injection** | Summarization-only pipeline, no generative freedom |

## 9. Cost Analysis

| Resource | Cost | Notes |
|----------|------|-------|
| **Training (Kaggle)** | Free | 30 hours GPU/week |
| **Training (Colab Pro)** | $10/mo | A100 access |
| **Inference (AWS g4dn.xlarge)** | $0.526/hr | T4 GPU |
| **Inference (CPU - t3.large)** | $0.0832/hr | Slow but cheap |
| **Docker Registry** | Free | GitHub Container Registry |

## 10. Future Improvements

1. **ONNX Export** - Convert model to ONNX for 30-50% faster inference
2. **Quantization** - INT8 quantization for 4x smaller model
3. **Streaming** - Real-time token-by-token generation via WebSocket
4. **Multi-language** - Fine-tune multilingual BART (mBART)
5. **Custom datasets** - Support user-uploaded training data
6. **A/B Testing** - Compare model versions in production
7. **MLflow Integration** - Full experiment tracking
