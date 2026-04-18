"""Unit tests for stage-3 data augmentation utilities."""

from textSummarizer.components.data_augmentation import DataAugmentation


def test_paraphrase_dialogue_returns_string() -> None:
    augmenter = DataAugmentation(augment_prob=1.0, seed=42)
    text = "I think this is good and I need help."
    out = augmenter.paraphrase_dialogue(text, intensity=1.0)

    assert isinstance(out, str)
    assert len(out) > 0


def test_inject_noise_preserves_length_bounds() -> None:
    augmenter = DataAugmentation(augment_prob=1.0, seed=42)
    text = "Alice and Bob will meet tomorrow at the office"
    out = augmenter.inject_noise(text, noise_level=0.2)

    assert isinstance(out, str)
    assert len(out.split()) == len(text.split())


def test_shuffle_dialogue_turns_keeps_boundary_turns() -> None:
    augmenter = DataAugmentation(augment_prob=1.0, seed=42)
    text = "Alice: Hello\nBob: Hi\nAlice: Can we meet?\nBob: Yes"
    turns_before = text.splitlines()

    out = augmenter.shuffle_dialogue_turns(text, max_shuffle=1.0)
    turns_after = out.splitlines()

    assert len(turns_after) == len(turns_before)
    assert turns_after[0] == turns_before[0]
    assert turns_after[-1] == turns_before[-1]


def test_remove_fillers_reduces_spoken_noise() -> None:
    augmenter = DataAugmentation(augment_prob=1.0, seed=42)
    text = "Um I think we should, you know, submit today"
    out = augmenter.remove_fillers(text)

    assert "Um" not in out
    assert "you know" not in out.lower()


def test_augment_dataset_batch_preserves_batch_size() -> None:
    augmenter = DataAugmentation(augment_prob=1.0, seed=42)
    batch = {
        "dialogue": [
            "Alice: Hi\nBob: Hello",
            "Alice: Ready?\nBob: Yes",
            "Alice: Thanks\nBob: Welcome",
        ],
        "summary": [
            "Alice and Bob greet each other.",
            "Alice and Bob confirm readiness.",
            "Alice thanks Bob.",
        ],
    }

    out = augmenter.augment_dataset_batch(batch)

    assert "dialogue" in out
    assert len(out["dialogue"]) == len(batch["dialogue"])
    assert len(out["summary"]) == len(batch["summary"])


def test_augment_probability_zero_keeps_text_unchanged() -> None:
    augmenter = DataAugmentation(augment_prob=0.0, seed=42)
    text = "Alice: Please send the file. Bob: Sure."

    out = augmenter.augment_text(text)

    assert out == text
