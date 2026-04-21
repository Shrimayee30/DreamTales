import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import TrainingConfig, parse_args
from data import load_tinystories_splits, tokenize_dataset


def resolve_dtype(config: TrainingConfig) -> torch.dtype:
    if torch.cuda.is_available() and config.use_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def default_target_modules(model_name: str) -> list[str]:
    lowered = model_name.lower()
    if "mistral" in lowered or "llama" in lowered or "tinyllama" in lowered:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return ["q_proj", "v_proj"]


def build_quantization_config(config: TrainingConfig) -> BitsAndBytesConfig | None:
    if not config.use_4bit:
        return None
    if not torch.cuda.is_available():
        print("4-bit quantization disabled because CUDA is not available.")
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=resolve_dtype(config),
    )


def load_model_and_tokenizer(config: TrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = build_quantization_config(config)
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = resolve_dtype(config)

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.config.use_cache = False

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    target_modules = config.target_modules or default_target_modules(config.model_name)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def build_training_arguments(config: TrainingConfig) -> TrainingArguments:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bf16 = torch.cuda.is_available() and config.use_bf16 and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        optim="adamw_torch",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=config.save_total_limit,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        remove_unused_columns=False,
        seed=config.seed,
    )


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_json(args.config)
    set_seed(config.seed)

    print(f"Loading tokenizer and base model: {config.model_name}")
    model, tokenizer = load_model_and_tokenizer(config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    print(f"Loading dataset: {config.dataset_name}")
    dataset = load_tinystories_splits(config)
    tokenized = tokenize_dataset(dataset, tokenizer, config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = build_training_arguments(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting DreamCore fine-tuning...")
    trainer.train()

    final_adapter_dir = Path(config.output_dir) / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    trainer.save_state()

    summary_path = Path(config.output_dir) / "training_config.json"
    summary_path.write_text(json.dumps(config.to_dict(), indent=2))

    print(f"Saved final adapter to: {final_adapter_dir}")
    print(f"Saved training config to: {summary_path}")


if __name__ == "__main__":
    main()
