import re

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from config import TrainingConfig


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def first_sentence(text: str) -> str:
    normalized = normalize_text(text)
    match = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)
    return match[0].strip() if match and match[0].strip() else normalized


def build_synthetic_prompt(story: str, max_words: int = 14) -> str:
    lead = first_sentence(story)
    words = lead.split()
    compact = " ".join(words[:max_words]).strip(" ,.;:!?")
    if not compact:
        compact = "a kind dream under the stars"
    return f"Write a short bedtime story for a child about {compact}."


def format_story_for_training(story: str, prompt_mode: str) -> str:
    normalized_story = normalize_text(story)
    if prompt_mode == "plain":
        return normalized_story

    prompt = build_synthetic_prompt(normalized_story)
    return f"Prompt: {prompt}\nStory:\n{normalized_story}"


def load_tinystories_splits(config: TrainingConfig) -> DatasetDict:
    dataset = load_dataset(config.dataset_name)
    train_split = dataset["train"]
    eval_split = dataset["validation"] if "validation" in dataset else dataset["train"]

    if config.max_train_samples is not None:
        train_split = train_split.select(range(min(config.max_train_samples, len(train_split))))
    if config.max_eval_samples is not None:
        eval_split = eval_split.select(range(min(config.max_eval_samples, len(eval_split))))

    return DatasetDict({"train": train_split, "validation": eval_split})


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: TrainingConfig,
) -> DatasetDict:
    text_field = config.dataset_text_field

    def format_batch(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        texts = []
        for story in batch[text_field]:
            formatted = format_story_for_training(story, config.prompt_mode)
            texts.append(formatted + tokenizer.eos_token)
        return {"text": texts}

    formatted = dataset.map(
        format_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Formatting TinyStories examples",
    )

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_seq_length,
        )

    return formatted.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing TinyStories",
    )
