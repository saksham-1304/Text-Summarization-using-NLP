# Low-Level Design (LLD) - Text Summarization System

## 1. Package Structure

```
Text-Summarization-NLP-Project/
│
├── app.py                          # FastAPI application entry point
├── main.py                         # Training pipeline orchestrator (CLI)
├── config/
│   └── config.yaml                 # Infrastructure/path configuration
├── params.yaml                     # Hyperparameters & training arguments
│
├── src/
│   └── textSummarizer/
│       ├── __init__.py
│       ├── constants/
│       │   └── __init__.py         # CONFIG_FILE_PATH, PARAMS_FILE_PATH
│       ├── entity/
│       │   └── __init__.py         # Frozen dataclass configs
│       ├── config/
│       │   └── configuration.py    # ConfigurationManager class
│       ├── components/
│       │   ├── data_ingestion.py   # Stage 1: Download from HF Hub
│       │   ├── data_validation.py  # Stage 2: Validate schema & quality
│       │   ├── data_transformation.py  # Stage 3: Tokenize
│       │   ├── model_trainer.py    # Stage 4: Fine-tune BART
│       │   └── model_evaluation.py # Stage 5: ROUGE evaluation
│       ├── pipeline/
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_validation.py
│       │   ├── stage_03_data_transformation.py
│       │   ├── stage_04_model_trainer.py
│       │   ├── stage_05_model_evaluation.py
│       │   └── prediction.py       # Inference pipeline
│       ├── utils/
│       │   └── common.py           # YAML, JSON, directory utilities
│       └── logging/
│           └── __init__.py         # Rotating file + console logger
│
├── templates/
│   └── index.html                  # Web UI template
├── static/
│   └── style.css                   # Web UI styles
├── tests/
│   ├── test_entity.py
│   ├── test_utils.py
│   ├── test_config.py
│   └── test_api.py
├── docs/
│   ├── HLD.md                      # High-Level Design
│   ├── LLD.md                      # This file
│   └── SYSTEM_DESIGN.md            # System Design Document
├── notebooks/
│   └── kaggle_training.ipynb       # Kaggle GPU training notebook
│
├── Dockerfile                      # Multi-stage Docker build
├── .github/workflows/ci.yml        # GitHub Actions CI
├── requirements.txt                # Pinned dependencies
├── setup.py                        # Package configuration
└── README.md                       # Project documentation
```

## 2. Class Diagrams

### 2.1 Entity Classes (Frozen Dataclasses)

```
┌─────────────────────────────┐
│    DataIngestionConfig      │
├─────────────────────────────┤
│ + root_dir: Path            │
│ + dataset_name: str         │
│ + local_data_dir: Path      │
└─────────────────────────────┘

┌─────────────────────────────┐
│   DataValidationConfig      │
├─────────────────────────────┤
│ + root_dir: Path            │
│ + status_file: Path         │
│ + local_data_dir: Path      │
│ + required_splits: List     │
│ + required_columns: Dict    │
└─────────────────────────────┘

┌─────────────────────────────┐
│  DataTransformationConfig   │
├─────────────────────────────┤
│ + root_dir: Path            │
│ + data_path: Path           │
│ + tokenizer_name: str       │
│ + max_input_length: int     │
│ + max_target_length: int    │
│ + text_column: str          │
│ + summary_column: str       │
└─────────────────────────────┘

┌─────────────────────────────┐
│    ModelTrainerConfig       │
├─────────────────────────────┤
│ + root_dir: Path            │
│ + data_path: Path           │
│ + model_ckpt: str           │
│ + num_train_epochs: int     │
│ + warmup_steps: int         │
│ + per_device_train_batch: int│
│ + per_device_eval_batch: int │
│ + weight_decay: float       │
│ + learning_rate: float      │
│ + fp16: bool                │
│ + ... (12 more fields)      │
└─────────────────────────────┘

┌─────────────────────────────┐
│   ModelEvaluationConfig     │
├─────────────────────────────┤
│ + root_dir: Path            │
│ + data_path: Path           │
│ + model_path: Path          │
│ + tokenizer_path: Path      │
│ + metric_file_name: Path    │
│ + batch_size: int           │
│ + max_input_length: int     │
│ + max_target_length: int    │
│ + text_column: str          │
│ + summary_column: str       │
└─────────────────────────────┘
```

### 2.2 Component Classes

```
┌────────────────────────────────────────────┐
│              DataIngestion                  │
├────────────────────────────────────────────┤
│ - config: DataIngestionConfig              │
├────────────────────────────────────────────┤
│ + download_dataset() -> None               │
│   Uses: datasets.load_dataset()            │
│   Saves: Arrow format to local_data_dir    │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│             DataValidation                 │
├────────────────────────────────────────────┤
│ - config: DataValidationConfig             │
├────────────────────────────────────────────┤
│ + validate_all_files_exist() -> bool       │
│ - _write_status(status, details) -> None   │
│   Checks: splits, columns, nulls, types   │
│   Output: JSON validation report           │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│           DataTransformation               │
├────────────────────────────────────────────┤
│ - config: DataTransformationConfig         │
│ - tokenizer: AutoTokenizer                 │
├────────────────────────────────────────────┤
│ + convert() -> None                        │
│ + convert_examples_to_features(batch) -> dict │
│   Tokenizes: dialogue (1024) + summary (128)  │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│             ModelTrainer                   │
├────────────────────────────────────────────┤
│ - config: ModelTrainerConfig               │
├────────────────────────────────────────────┤
│ + train() -> None                          │
│   Features: fp16, grad checkpointing,     │
│   early stopping, best model selection    │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│            ModelEvaluation                 │
├────────────────────────────────────────────┤
│ - config: ModelEvaluationConfig            │
├────────────────────────────────────────────┤
│ + evaluate() -> dict                       │
│ + calculate_metric_on_test_ds() -> dict    │
│ - _generate_batch_sized_chunks() -> gen    │
│   Metrics: ROUGE-1/2/L/Lsum               │
│   Output: CSV + JSON                       │
└────────────────────────────────────────────┘
```

### 2.3 ConfigurationManager

```
┌────────────────────────────────────────────┐
│          ConfigurationManager              │
├────────────────────────────────────────────┤
│ - config: ConfigBox (config.yaml)          │
│ - params: ConfigBox (params.yaml)          │
├────────────────────────────────────────────┤
│ + get_data_ingestion_config()              │
│ + get_data_validation_config()             │
│ + get_data_transformation_config()         │
│ + get_model_trainer_config()               │
│ + get_model_evaluation_config()            │
└────────────────────────────────────────────┘
```

## 3. Sequence Diagrams

### 3.1 Training Flow

```
main.py          Pipeline          Component         ConfigMgr         FileSystem
   │                │                  │                  │                 │
   │── run_stage(1) │                  │                  │                 │
   │                │── __init__() ───►│                  │                 │
   │                │── main() ───────►│                  │                 │
   │                │                  │── get_config() ──►                 │
   │                │                  │◄─── config ──────│                 │
   │                │                  │── download() ────────────────────► │
   │                │                  │◄──── saved ──────────────────────  │
   │                │◄─── done ────────│                  │                 │
   │                │                  │                  │                 │
   │── run_stage(2) │                  │                  │                 │
   │                │── main() ───────►│                  │                 │
   │                │                  │── validate() ────────────────────► │
   │                │                  │◄──── report ─────────────────────  │
   │                │◄─── bool ────────│                  │                 │
   │    ...         │    ...           │    ...           │      ...        │
```

### 3.2 Inference Flow

```
Client          FastAPI           PredictionPipeline       BART Model
   │                │                      │                    │
   │── POST /predict│                      │                    │
   │                │── predict(text) ────►│                    │
   │                │                      │── tokenize() ────►│
   │                │                      │── generate() ────►│
   │                │                      │◄── token_ids ─────│
   │                │                      │── decode() ──────►│
   │                │◄── summary ──────────│                    │
   │◄── JSON ───────│                      │                    │
```

## 4. Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **Factory** | ConfigurationManager | Creates typed config objects from YAML |
| **Pipeline** | Stage orchestration | Sequential execution of independent stages |
| **Singleton-like** | PredictionPipeline | Loaded once at app startup via lifespan |
| **Strategy** | Component constructors | Accept config to change behavior |
| **Template Method** | Pipeline stages | Common main() interface, different implementations |

## 5. Error Handling Strategy

```
Level 1: Component methods
    └── Try/except with logger.exception()
    └── Raise specific exceptions (RuntimeError, ValueError)

Level 2: Pipeline stages
    └── Catch component exceptions
    └── Write status files for debugging

Level 3: main.py orchestrator
    └── Catch per-stage errors
    └── Log and sys.exit(1) on failure

Level 4: FastAPI endpoints
    └── HTTPException with status codes
    └── 503 for model not loaded
    └── 500 for prediction errors
    └── Pydantic validation for input
```

## 6. Configuration Separation

```
config.yaml (WHAT to process)          params.yaml (HOW to process)
├── Paths (root_dir, data_path)        ├── TrainingArguments
├── Dataset name (samsum)              │   ├── epochs, batch_size
├── Model checkpoint name              │   ├── learning_rate
├── Output file paths                  │   └── fp16, warmup_steps
└── Column names per split             ├── DataTransformation
                                       │   ├── max_input_length
                                       │   └── text_column
                                       └── ModelEvaluation
                                           ├── batch_size
                                           └── max_target_length
```
