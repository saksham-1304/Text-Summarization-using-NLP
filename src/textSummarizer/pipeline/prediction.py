"""Prediction Pipeline for inference.

Loads the fine-tuned model and generates summaries for input dialogues.
Supports both single and batch predictions.
"""

from transformers import AutoTokenizer, pipeline as hf_pipeline
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger


class PredictionPipeline:
    """Handles loading the model and running inference."""

    def __init__(self) -> None:
        self.config = ConfigurationManager().get_model_evaluation_config()
        logger.info(f"Loading model from: {self.config.model_path}")
        logger.info(f"Loading tokenizer from: {self.config.tokenizer_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.tokenizer_path)
        )
        self.pipe = hf_pipeline(
            "summarization",
            model=str(self.config.model_path),
            tokenizer=self.tokenizer,
        )
        self.gen_kwargs = {
            "length_penalty": 0.8,
            "num_beams": 8,
            "max_length": self.config.max_target_length,
        }
        logger.info("Prediction pipeline initialized")

    def predict(self, text: str) -> str:
        """Generate a summary for the given text.

        Args:
            text: Input dialogue or text to summarize.

        Returns:
            Generated summary string.
        """
        if not text or not text.strip():
            return ""

        output = self.pipe(text, **self.gen_kwargs)[0]["summary_text"]
        logger.info(f"Generated summary ({len(output)} chars) for input ({len(text)} chars)")
        return output

    def predict_batch(self, texts: list) -> list:
        """Generate summaries for a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of generated summary strings.
        """
        results = self.pipe(texts, **self.gen_kwargs)
        return [r["summary_text"] for r in results]