"""Prediction Pipeline for inference.

Loads the fine-tuned model and generates summaries for input dialogues.
Supports both single and batch predictions.
Uses model.generate() directly — compatible with transformers 4.x and 5.x.

Resource management:
  - Model loaded once on initialization
  - Automatic cleanup via context manager or __del__
  - GPU memory released on shutdown
"""

import gc
from pathlib import Path
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger
from textSummarizer.utils.device import get_device_and_dtype

# Suppress noisy generation-config warning from transformers 5.x
warnings.filterwarnings("ignore", message="Please make sure the generation config")
warnings.filterwarnings("ignore", message="Please make sure the config includes `forced_bos_token_id=0`")


class PredictionPipeline:
    """Handles loading the model and running inference with proper resource cleanup."""

    def __init__(self) -> None:
        self.config = ConfigurationManager().get_model_evaluation_config()
        logger.info(f"Loading tokenizer from: {self.config.tokenizer_path}")
        logger.info(f"Loading model from: {self.config.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.tokenizer_path)
        )
        
        # Use centralized device detection
        self.device, model_dtype = get_device_and_dtype()

        load_kwargs = {
            "low_cpu_mem_usage": True,
            "dtype": model_dtype,
        }

        # Load model with device-aware strategy and proper error handling
        model_path = str(self.config.model_path)
        logger.info(f"Loading model from {model_path} on device={self.device} with dtype={model_dtype}")
        
        try:
            if self.device.type == "cuda":
                offload_dir = Path("offload_cache")
                offload_dir.mkdir(parents=True, exist_ok=True)
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        offload_folder=str(offload_dir),
                        **load_kwargs,
                    )
                    logger.info("✓ Model loaded with device mapping on CUDA")
                except (RuntimeError, torch.cuda.OutOfMemoryError) as mem_err:
                    logger.warning(
                        f"Device mapping OOM: {mem_err}. Falling back to single-device..."
                    )
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_path,
                        device_map=None,
                        **load_kwargs,
                    ).to(self.device)
                    logger.info(f"✓ Model loaded on single device={self.device}")
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    device_map=None,
                    **load_kwargs,
                ).to(self.device)
                logger.info(f"✓ Model loaded on device={self.device} (dtype={model_dtype})")
        
        except FileNotFoundError as e:
            logger.error(f"❌ FATAL: Model not found at {model_path}")
            logger.error("Train model first: python main.py")
            raise FileNotFoundError(f"Model not found: {model_path}") from e
        
        except (ValueError, TypeError) as e:
            logger.error(f"❌ FATAL: Model config error: {e}")
            raise RuntimeError(f"Model corrupted/incompatible: {e}") from e
        
        except Exception as e:
            logger.error(f"❌ FATAL: Unexpected load error: {e}", exc_info=True)
            raise RuntimeError(f"Unrecoverable model error: {e}") from e

        self.model.eval()

        # Avoid warning noise from generation config without forcing incompatible configs.
        if hasattr(self.model, "generation_config"):
            if getattr(self.model.generation_config, "forced_bos_token_id", None) is not None:
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
        logger.info(f"✓ Prediction pipeline ready (device={self.device}, dtype={model_dtype})")

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
            text: Input dialogue or text to summarize (non-empty string).
            max_length: Maximum new tokens for summary. Defaults to config value.
                       Must be in [16, 512] if provided.

        Returns:
            Generated summary string (may be empty for invalid input).
        
        Raises:
            ValueError: If text is invalid (not string, empty, etc).
            TypeError: If max_length is not int.
        """
        # Input validation
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text)}")
        
        if not text or not text.strip():
            logger.warning("Empty input text")
            return ""
        
        if len(text.strip()) < 10:
            logger.warning(f"Input text too short: {len(text)} chars (min 10)")
        
        if max_length is not None:
            if not isinstance(max_length, int):
                raise TypeError(f"max_length must be int, got {type(max_length)}")
            if not (16 <= max_length <= 512):
                raise ValueError(f"max_length must be in [16, 512], got {max_length}")

        results = self._generate([text], max_length=max_length)
        summary = results[0]
        logger.info(f"Generated summary ({len(summary)} chars) for input ({len(text)} chars)")
        return summary

    def predict_batch(self, texts: list, max_length: int = None) -> list:
        """Generate summaries for a batch of texts.

        Args:
            texts: List of input texts to summarize (non-empty list).
            max_length: Maximum new tokens for each summary. Defaults to config value.
                       Must be in [16, 512] if provided.

        Returns:
            List of generated summary strings.
        
        Raises:
            ValueError: If texts is invalid (empty, contains non-strings, etc).
            TypeError: If inputs are wrong type.
        """
        # Input validation
        if not isinstance(texts, list):
            raise TypeError(f"texts must be list, got {type(texts)}")
        
        if len(texts) == 0:
            logger.warning("Empty batch")
            return []
        
        if len(texts) > 100:
            logger.warning(f"Large batch size: {len(texts)}")
        
        # Validate each text
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{i}] must be str, got {type(text)}")
            if not text or not text.strip():
                raise ValueError(f"texts[{i}] is empty or whitespace-only")
        
        if max_length is not None:
            if not isinstance(max_length, int):
                raise TypeError(f"max_length must be int, got {type(max_length)}")
            if not (16 <= max_length <= 512):
                raise ValueError(f"max_length must be in [16, 512], got {max_length}")

        return self._generate(texts, max_length=max_length)

    def cleanup(self) -> None:
        """Release model memory and GPU resources.
        
        Should be called before application shutdown to prevent memory leaks.
        Safe to call multiple times.
        """
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
                logger.info("✓ Model unloaded from memory")
            
            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                del self.tokenizer
                logger.info("✓ Tokenizer unloaded")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✓ GPU cache cleared")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def __del__(self) -> None:
        """Cleanup on garbage collection (safety net)."""
        try:
            self.cleanup()
        except Exception:
            pass  # Silently ignore errors during garbage collection