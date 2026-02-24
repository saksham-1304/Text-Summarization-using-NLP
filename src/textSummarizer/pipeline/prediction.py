"""Prediction Pipeline for inference.

Loads the fine-tuned model and generates summaries for input dialogues.
Supports both single and batch predictions.
Uses model.generate() directly — compatible with transformers 4.x and 5.x.
"""

import gc
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger

# Suppress noisy generation-config warning from transformers 5.x
warnings.filterwarnings("ignore", message="Please make sure the generation config")


class PredictionPipeline:
    """Handles loading the model and running inference."""

    def __init__(self) -> None:
        self.config = ConfigurationManager().get_model_evaluation_config()
        logger.info(f"Loading tokenizer from: {self.config.tokenizer_path}")
        logger.info(f"Loading model from: {self.config.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.tokenizer_path)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Memory-efficient loading: use accelerate's device_map for automatic
        # memory management; offloads layers to disk if RAM is tight.
        import os
        offload_dir = "offload_cache"
        os.makedirs(offload_dir, exist_ok=True)

        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.config.model_path),
                low_cpu_mem_usage=True,
                dtype=torch.float16,
                device_map="auto",
                offload_folder=offload_dir,
            )
            logger.info(f"Model loaded with device_map=auto fp16 (device={self.device})")
        except Exception:
            # Fallback: plain load without device_map (requires sufficient RAM)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.config.model_path),
                low_cpu_mem_usage=True,
                dtype=torch.float16,
            ).to(self.device)
            logger.info(f"Model loaded fp16 (fallback, device={self.device})")

        self.model.eval()
        # Suppress "forced_bos_token_id" warning from the saved generation config
        self.model.generation_config.forced_bos_token_id = 0

        self.default_max_length = self.config.max_target_length
        # Use greedy decoding (num_beams=1) to reduce repetition, but keep
        # a small length penalty and n-gram ban to avoid loops.
        self.gen_kwargs = {
            "num_beams": 1,
            "length_penalty": 0.8,
            "no_repeat_ngram_size": 5,
            "early_stopping": True,
            "forced_bos_token_id": 0,
        }
        logger.info(f"Prediction pipeline initialized (device={self.device})")

    def _generate(self, texts: list, max_length: int = None) -> list:
        """Tokenize, generate, and decode a list of texts."""
        max_len = max_length or self.default_max_length

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True,
        )

        # When device_map is used the model spans multiple devices; put inputs
        # on the same device as the first model parameter.
        try:
            input_device = next(self.model.parameters()).device
        except StopIteration:
            input_device = self.device

        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)

        gc.collect()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_len,
                **self.gen_kwargs,
            )
        gc.collect()

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def predict(self, text: str, max_length: int = None) -> str:
        """Generate a summary for the given text.

        Args:
            text: Input dialogue or text to summarize.
            max_length: Maximum new tokens for the summary. Defaults to config value.

        Returns:
            Generated summary string.
        """
        if not text or not text.strip():
            return ""

        results = self._generate([text], max_length=max_length)
        summary = results[0]
        logger.info(f"Generated summary ({len(summary)} chars) for input ({len(text)} chars)")
        return summary

    def predict_batch(self, texts: list, max_length: int = None) -> list:
        """Generate summaries for a batch of texts.

        Args:
            texts: List of input texts.
            max_length: Maximum new tokens for summaries. Defaults to config value.

        Returns:
            List of generated summary strings.
        """
        if not texts:
            return []
        return self._generate(texts, max_length=max_length)