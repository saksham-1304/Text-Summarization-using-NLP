"""Advanced data augmentation utilities for dialogue summarization.

The goal is to increase training diversity without changing ground-truth
semantics too aggressively.
"""

import random
import re
from typing import Dict, List

from textSummarizer.logging import logger


class DataAugmentation:
    """Augmentation strategies for dialogue summarization."""

    PRESERVE_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "and", "or", "but", "if",
        "then", "else", "for", "while", "at", "to", "of", "in", "on", "by",
        "from", "up", "about", "as", "into", "through", "during", "before", "after",
    }

    FILLER_PATTERN = re.compile(
        r"\b(?:uh+|um+|hmm+|you know|kind of|sort of|like)\b[, ]*",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        augment_prob: float = 0.25,
        enable_augmentation: bool = True,
        seed: int = 42,
        noise_token: str = "[NOISE]",
    ) -> None:
        self.augment_prob = max(0.0, min(1.0, augment_prob))
        self.enable_augmentation = enable_augmentation
        self.noise_token = noise_token
        self.rng = random.Random(seed)

    @staticmethod
    def _split_turns(text: str) -> List[str]:
        """Split dialogue text into conversation turns."""
        if not text:
            return []

        if "\n" in text:
            turns = [t.strip() for t in text.splitlines() if t.strip()]
            return turns

        turns = re.split(r"(?=(?:[A-Za-z][A-Za-z0-9_ ]{0,24}:))", text)
        return [t.strip() for t in turns if t.strip()]

    def paraphrase_dialogue(self, text: str, intensity: float = 0.15) -> str:
        """Apply light lexical paraphrasing."""
        if not text:
            return text

        replacements = [
            ("can't", "cannot"),
            ("won't", "will not"),
            ("okay", "alright"),
            ("thanks", "thank you"),
            ("need", "require"),
            ("want", "would like"),
            ("help", "assist"),
            ("good", "great"),
            ("bad", "poor"),
            ("think", "believe"),
            ("quick", "fast"),
            ("small", "tiny"),
            ("big", "large"),
        ]

        result = text
        for source, target in replacements:
            if self.rng.random() < intensity:
                pattern = rf"\b{re.escape(source)}\b"
                result = re.sub(pattern, target, result, flags=re.IGNORECASE)

        return result

    def inject_noise(self, text: str, noise_level: float = 0.05) -> str:
        """Mask a small subset of non-critical tokens."""
        tokens = text.split()
        if not tokens:
            return text

        candidates = []
        for idx, token in enumerate(tokens):
            normalized = re.sub(r"[^a-zA-Z]", "", token).lower()
            if normalized and normalized not in self.PRESERVE_WORDS:
                candidates.append(idx)

        if not candidates:
            return text

        replace_count = max(1, int(len(candidates) * noise_level))
        chosen = self.rng.sample(candidates, min(replace_count, len(candidates)))
        for idx in chosen:
            tokens[idx] = self.noise_token

        return " ".join(tokens)

    def shuffle_dialogue_turns(self, text: str, max_shuffle: float = 0.2) -> str:
        """Shuffle a subset of middle turns while preserving start/end context."""
        turns = self._split_turns(text)
        if len(turns) < 4:
            return text

        middle = turns[1:-1]
        if len(middle) < 2:
            return text

        sample_size = max(2, int(round(len(middle) * max_shuffle)))
        sample_size = min(sample_size, len(middle))

        selected_idx = self.rng.sample(range(len(middle)), sample_size)
        selected_turns = [middle[idx] for idx in selected_idx]
        self.rng.shuffle(selected_turns)
        for out_idx, middle_idx in enumerate(selected_idx):
            middle[middle_idx] = selected_turns[out_idx]

        separator = "\n" if "\n" in text else " "
        return separator.join([turns[0], *middle, turns[-1]])

    def remove_fillers(self, text: str) -> str:
        """Remove conversational filler words to improve robustness."""
        if not text:
            return text
        return re.sub(self.FILLER_PATTERN, "", text).strip()

    def augment_text(self, dialogue: str) -> str:
        """Apply one random augmentation operation."""
        if not self.enable_augmentation or self.rng.random() > self.augment_prob:
            return dialogue

        operation = self.rng.choice(["paraphrase", "noise", "shuffle", "de_filler"])
        if operation == "paraphrase":
            return self.paraphrase_dialogue(dialogue, intensity=0.15)
        if operation == "noise":
            return self.inject_noise(dialogue, noise_level=0.05)
        if operation == "shuffle":
            return self.shuffle_dialogue_turns(dialogue, max_shuffle=0.2)
        return self.remove_fillers(dialogue)

    def augment_dataset_batch(self, examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Apply augmentation over a batched dataset mapping call."""
        if not self.enable_augmentation:
            return examples

        dialogue_col = "dialogue" if "dialogue" in examples else "text"
        updated = examples.copy()
        updated[dialogue_col] = [self.augment_text(d) for d in examples[dialogue_col]]
        return updated


def get_augmentation_strategy(
    augment_prob: float = 0.25,
    enable_augmentation: bool = True,
    seed: int = 42,
) -> DataAugmentation:
    """Factory helper for stage-3 preprocessing."""
    logger.info(
        "Creating data augmentation strategy: "
        f"enabled={enable_augmentation}, prob={augment_prob}, seed={seed}"
    )
    return DataAugmentation(
        augment_prob=augment_prob,
        enable_augmentation=enable_augmentation,
        seed=seed,
    )
