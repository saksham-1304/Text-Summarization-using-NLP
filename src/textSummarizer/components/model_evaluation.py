"""Stage 5: Model Evaluation Component.

Evaluates the fine-tuned model on the test split using:
  - ROUGE-1: Unigram overlap (precision, recall, F1)
  - ROUGE-2: Bigram overlap (captures phrase-level similarity)
  - ROUGE-L: Longest common subsequence
  - ROUGE-Lsum: Sentence-level ROUGE-L (better for multi-sentence summaries)

Uses the modern `evaluate` library (replacement for deprecated `datasets.load_metric`).
Saves metrics to both CSV and JSON for downstream consumption.
"""

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate
from textSummarizer.logging import logger
from textSummarizer.entity import ModelEvaluationConfig


class ModelEvaluation:
    """Evaluates a trained summarization model on test data."""

    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        batch_size: int = 8,
        device: str = "cpu",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
    ) -> dict:
        """Calculate ROUGE metrics on the test dataset.

        Args:
            dataset: HuggingFace dataset split.
            metric: evaluate.Metric instance for ROUGE.
            model: Loaded seq2seq model.
            tokenizer: Loaded tokenizer.
            batch_size: Inference batch size.
            device: Device to run inference on.
            generation_kwargs: Decoding kwargs passed to `model.generate`.
            max_samples: Optional cap for number of examples to evaluate.

        Returns:
            Dict of ROUGE metric scores.
        """
        text_column = self.config.text_column
        summary_column = self.config.summary_column

        if max_samples is not None:
            sample_count = min(max_samples, len(dataset))
            dataset = dataset.select(range(sample_count))

        if generation_kwargs is None:
            generation_kwargs = {
                "length_penalty": self.config.default_length_penalty,
                "num_beams": self.config.default_num_beams,
                "no_repeat_ngram_size": self.config.default_no_repeat_ngram_size,
            }

        total_rows = len(dataset)
        for start_idx in tqdm(
            range(0, total_rows, batch_size),
            total=(total_rows + batch_size - 1) // batch_size,
            desc="Evaluating",
        ):
            end_idx = min(start_idx + batch_size, total_rows)
            article_batch = dataset[text_column][start_idx:end_idx]
            target_batch = dataset[summary_column][start_idx:end_idx]

            inputs = tokenizer(
                article_batch,
                max_length=self.config.max_input_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                max_length=self.config.max_target_length,
                **generation_kwargs,
            )

            decoded_summaries = [
                tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for s in summaries
            ]

            # Clean up decoded text
            decoded_summaries = [d.strip() for d in decoded_summaries]

            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        score = metric.compute()
        return score

    @staticmethod
    def _normalize_rouge_scores(score: dict) -> dict:
        """Convert metric output to plain float dict for serialization."""
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = {}
        for rn in rouge_names:
            value = score[rn]
            if hasattr(value, "mid"):
                rouge_dict[rn] = value.mid.fmeasure
            else:
                rouge_dict[rn] = float(value)
        return rouge_dict

    def _run_decoding_sweep(
        self,
        dataset,
        model,
        tokenizer,
        device: str,
    ) -> dict:
        """Tune decoding hyperparameters on validation split and return best kwargs."""
        val_split = dataset["validation"]
        combos = list(
            itertools.product(
                self.config.decoding_num_beams,
                self.config.decoding_length_penalties,
                self.config.decoding_no_repeat_ngram_sizes,
            )
        )

        selection_metric = self.config.decoding_selection_metric
        if selection_metric not in {"rouge1", "rouge2", "rougeL", "rougeLsum"}:
            logger.warning(
                f"Unknown decoding_selection_metric={selection_metric}; using rougeLsum"
            )
            selection_metric = "rougeLsum"

        logger.info(
            "Running decoding sweep on validation split: "
            f"{len(combos)} combinations, max_samples={self.config.decoding_sweep_max_samples}"
        )

        sweep_rows = []
        best_kwargs = {
            "num_beams": self.config.default_num_beams,
            "length_penalty": self.config.default_length_penalty,
            "no_repeat_ngram_size": self.config.default_no_repeat_ngram_size,
        }
        best_score = float("-inf")
        best_rouge1 = float("-inf")

        for idx, (num_beams, length_penalty, no_repeat_ngram_size) in enumerate(combos, start=1):
            generation_kwargs = {
                "num_beams": int(num_beams),
                "length_penalty": float(length_penalty),
                "no_repeat_ngram_size": int(no_repeat_ngram_size),
            }

            rouge_metric = evaluate.load("rouge")
            raw_score = self.calculate_metric_on_test_ds(
                val_split,
                rouge_metric,
                model,
                tokenizer,
                batch_size=self.config.batch_size,
                device=device,
                generation_kwargs=generation_kwargs,
                max_samples=self.config.decoding_sweep_max_samples,
            )
            score = self._normalize_rouge_scores(raw_score)
            selection_value = score[selection_metric]

            row = {
                "trial": idx,
                "num_beams": generation_kwargs["num_beams"],
                "length_penalty": generation_kwargs["length_penalty"],
                "no_repeat_ngram_size": generation_kwargs["no_repeat_ngram_size"],
                **score,
                "selection_metric": selection_metric,
                "selection_value": selection_value,
            }
            sweep_rows.append(row)

            if (
                selection_value > best_score
                or (selection_value == best_score and score["rouge1"] > best_rouge1)
            ):
                best_score = selection_value
                best_rouge1 = score["rouge1"]
                best_kwargs = generation_kwargs

        sweep_df = pd.DataFrame(sweep_rows).sort_values(
            by=["selection_value", "rouge1"], ascending=False
        )

        metrics_path = Path(self.config.metric_file_name)
        output_dir = metrics_path.parent
        sweep_csv = output_dir / "decoding_sweep_results.csv"
        sweep_json = output_dir / "decoding_sweep_results.json"
        best_json = output_dir / "best_decoding_config.json"

        sweep_df.to_csv(sweep_csv, index=False)
        with open(sweep_json, "w") as f:
            json.dump(sweep_rows, f, indent=2)
        with open(best_json, "w") as f:
            json.dump(
                {
                    "selection_metric": selection_metric,
                    "best_generation_kwargs": best_kwargs,
                    "best_selection_value": best_score,
                },
                f,
                indent=2,
            )

        logger.info(f"Decoding sweep saved: {sweep_csv}")
        logger.info(f"Best decoding config: {best_kwargs}")

        return best_kwargs

    def evaluate(self) -> dict:
        """Run full evaluation pipeline.

        Workflow:
        1. Load model and tokenizer from saved paths
        2. Load tokenized test dataset
        3. Compute ROUGE scores
        4. Save metrics to CSV and JSON
        5. Log results

        Returns:
            Dict of metric name -> score.
        """
        # Device selection
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Evaluation device: {device}")

        # Load model and tokenizer
        logger.info(f"Loading tokenizer from: {self.config.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_path))

        logger.info(f"Loading model from: {self.config.model_path}")
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self.config.model_path)).to(device)
        model.eval()

        # Load dataset
        logger.info(f"Loading dataset from: {self.config.data_path}")
        dataset = load_from_disk(str(self.config.data_path))

        # Load ROUGE metric using the modern `evaluate` library
        rouge_metric = evaluate.load("rouge")

        generation_kwargs = {
            "num_beams": self.config.default_num_beams,
            "length_penalty": self.config.default_length_penalty,
            "no_repeat_ngram_size": self.config.default_no_repeat_ngram_size,
        }

        if self.config.enable_decoding_sweep:
            generation_kwargs = self._run_decoding_sweep(
                dataset=dataset,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )

        # Evaluate on test set
        logger.info(
            f"Evaluating on {len(dataset['test'])} test examples "
            f"(batch_size={self.config.batch_size}) with generation={generation_kwargs}"
        )

        score = self.calculate_metric_on_test_ds(
            dataset["test"],
            rouge_metric,
            model,
            tokenizer,
            batch_size=self.config.batch_size,
            device=device,
            generation_kwargs=generation_kwargs,
        )

        # Extract scores
        rouge_dict = self._normalize_rouge_scores(score)
        rouge_dict["num_beams"] = generation_kwargs["num_beams"]
        rouge_dict["length_penalty"] = generation_kwargs["length_penalty"]
        rouge_dict["no_repeat_ngram_size"] = generation_kwargs["no_repeat_ngram_size"]

        # Log results
        logger.info("=" * 50)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 50)
        for name, value in rouge_dict.items():
            logger.info(f"  {name}: {value:.4f}")
        logger.info("=" * 50)

        # Save to CSV
        df = pd.DataFrame(rouge_dict, index=["bart-samsum"])
        csv_path = str(self.config.metric_file_name)
        df.to_csv(csv_path, index_label="model")
        logger.info(f"Metrics saved to CSV: {csv_path}")

        # Save to JSON
        json_path = csv_path.replace(".csv", ".json")
        with open(json_path, "w") as f:
            json.dump(rouge_dict, f, indent=2)
        logger.info(f"Metrics saved to JSON: {json_path}")

        return rouge_dict

        

