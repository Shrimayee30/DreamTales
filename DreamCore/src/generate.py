import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a story with a DreamCore LoRA adapter.")
    parser.add_argument("--base-model", required=True, help="Base model used during fine-tuning.")
    parser.add_argument("--adapter-path", required=True, help="Path to the saved LoRA adapter.")
    parser.add_argument("--prompt", required=True, help="Story prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    prompt = f"Prompt: Write a short bedtime story for a child about {args.prompt}.\nStory:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded)


if __name__ == "__main__":
    main()
