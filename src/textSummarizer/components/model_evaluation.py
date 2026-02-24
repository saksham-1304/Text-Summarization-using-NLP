"""Stage 5: Model Evaluation Component.

Evaluates the fine-tuned model on the test split using:
  - ROUGE-1: Unigram overlap (precision, recall, F1)
  - ROUGE-2: Bigram overlap (captures phrase-level similarity)
  - ROUGE-L: Longest common subsequence
  - ROUGE-Lsum: Sentence-level ROUGE-L (better for multi-sentence summaries)

Uses the modern `evaluate` library (replacement for deprecated `datasets.load_metric`).
Saves metrics to both CSV and JSON for downstream consumption.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
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

    def _generate_batch_sized_chunks(self, list_of_elements: list, batch_size: int):
        """Yield successive batch-sized chunks from a list.

        Args:
            list_of_elements: The list to chunk.
            batch_size: Size of each chunk.

        Yields:
            Chunks of the input list.
        """
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        batch_size: int = 8,
        device: str = "cpu",
    ) -> dict:
        """Calculate ROUGE metrics on the test dataset.

        Args:
            dataset: HuggingFace dataset split.
            metric: evaluate.Metric instance for ROUGE.
            model: Loaded seq2seq model.
            tokenizer: Loaded tokenizer.
            batch_size: Inference batch size.
            device: Device to run inference on.

        Returns:
            Dict of ROUGE metric scores.
        """
        text_column = self.config.text_column
        summary_column = self.config.summary_column

        article_batches = list(
            self._generate_batch_sized_chunks(dataset[text_column], batch_size)
        )
        target_batches = list(
            self._generate_batch_sized_chunks(dataset[summary_column], batch_size)
        )

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches),
            total=len(article_batches),
            desc="Evaluating",
        ):
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
                length_penalty=0.8,
                num_beams=1,
                no_repeat_ngram_size=5,
                max_length=self.config.max_target_length,
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

        # Evaluate on test set
        logger.info(
            f"Evaluating on {len(dataset['test'])} test examples "
            f"(batch_size={self.config.batch_size})"
        )

        score = self.calculate_metric_on_test_ds(
            dataset["test"],
            rouge_metric,
            model,
            tokenizer,
            batch_size=self.config.batch_size,
            device=device,
        )

        # Extract scores
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = {}
        for rn in rouge_names:
            # evaluate library returns float directly in newer versions
            value = score[rn]
            if hasattr(value, "mid"):
                rouge_dict[rn] = value.mid.fmeasure
            else:
                rouge_dict[rn] = float(value)

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

        

