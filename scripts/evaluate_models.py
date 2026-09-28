"""
Evaluation cell: run 10 held-out 311 complaints through the base model, the
LoRA/QLoRA-tuned (SFT) model, and the DPO-tuned model, and print the three
outputs side by side.

"Held-out" means these complaints are freshly generated with a seed distinct
from the SFT training seed (42, see generate_synthetic_complaints.py) and the
DPO preference seed (123, see generate_dpo_pairs.py), so none of them were
seen during either training stage. --no-address-rate defaults higher than
training's 0.15 so a handful of no-address cases reliably show up even in
only 10 samples -- exactly the case the DPO pass targeted.

Only ONE base model is loaded, with both LoRA adapters ("sft" and "dpo")
attached to the same PeftModel; base-only output uses peft's
`disable_adapter()` context (bypasses both adapters, running the frozen
pretrained weights), so there's a single ~500MB model in memory rather than
three separate copies.

Usage (paste into a Colab cell, or run with %run for a bonus rendered HTML
table; a plain-text table always prints either way):
    %run scripts/evaluate_models.py \
        --sft-adapter outputs/qlora-gpt2-311/adapter \
        --dpo-adapter outputs/dpo-gpt2-311/adapter
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_synthetic_complaints import generate_dataset
from prompting import build_prompt


def load_models(model_name: str, sft_adapter: str, dpo_adapter: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    base.config.pad_token_id = tokenizer.pad_token_id
    base.to(device)

    model = PeftModel.from_pretrained(base, sft_adapter, adapter_name="sft")
    model.load_adapter(dpo_adapter, adapter_name="dpo")
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_evaluation(model, tokenizer, complaints: list[dict], device: str, max_new_tokens: int) -> list[dict]:
    rows = []
    for rec in complaints:
        prompt = build_prompt(rec["complaint"])

        with model.disable_adapter():
            base_out = generate(model, tokenizer, prompt, device, max_new_tokens)

        model.set_adapter("sft")
        lora_out = generate(model, tokenizer, prompt, device, max_new_tokens)

        model.set_adapter("dpo")
        dpo_out = generate(model, tokenizer, prompt, device, max_new_tokens)

        rows.append(
            {
                "complaint": rec["complaint"],
                "expected": json.dumps(rec["ticket"], ensure_ascii=False),
                "base": base_out,
                "lora_sft": lora_out,
                "dpo": dpo_out,
            }
        )
    return rows


def _wrap_lines(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def print_table_ascii(rows: list[dict], col_width: int = 34) -> None:
    headers = ("BASE (untuned)", "LORA/SFT", "DPO")
    header_line = " | ".join(h.ljust(col_width) for h in headers)
    rule = "-" * len(header_line)

    for i, row in enumerate(rows, 1):
        print("=" * len(header_line))
        print(f"[{i}] COMPLAINT: {row['complaint']}")
        print(f"    EXPECTED : {row['expected']}")
        print(rule)
        print(header_line)
        print(rule)

        cols = [_wrap_lines(row["base"], col_width), _wrap_lines(row["lora_sft"], col_width), _wrap_lines(row["dpo"], col_width)]
        height = max(len(c) for c in cols)
        for r in range(height):
            cells = [c[r] if r < len(c) else "" for c in cols]
            print(" | ".join(cell.ljust(col_width) for cell in cells))
    print("=" * len(header_line))


def maybe_display_html_table(rows: list[dict]) -> None:
    """Best-effort: also render a nicer HTML table if actually running inside
    a live notebook kernel (not just a plain `python script.py` subprocess)."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
            return
        import pandas as pd
        from IPython.display import display

        df = pd.DataFrame(rows)
        pd.set_option("display.max_colwidth", None)
        display(df.style.set_properties(**{"white-space": "pre-wrap", "text-align": "left", "vertical-align": "top"}))
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--sft-adapter", default="outputs/qlora-gpt2-311/adapter")
    parser.add_argument("--dpo-adapter", default="outputs/dpo-gpt2-311/adapter")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument(
        "--held-out-seed",
        type=int,
        default=999,
        help="distinct from the SFT (42) and DPO (123) dataset seeds, so none of these were trained on",
    )
    parser.add_argument(
        "--no-address-rate",
        type=float,
        default=0.3,
        help="higher than training's 0.15, so a few no-address cases reliably show up in only 10 samples",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--out", type=str, default="", help="optional path to also save raw results as JSONL")
    # parse_known_args so this also runs unmodified inside Jupyter/Colab.
    args, _unknown = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: no GPU detected, running eval on CPU (correct, just slower).")

    complaints = generate_dataset(args.n, args.no_address_rate, args.held_out_seed)

    print(f"Loading base model ({args.model_name}) + sft/dpo adapters onto {device}...")
    model, tokenizer = load_models(args.model_name, args.sft_adapter, args.dpo_adapter, device)

    print(f"Generating outputs for {len(complaints)} held-out complaints...")
    rows = run_evaluation(model, tokenizer, complaints, device, args.max_new_tokens)

    print_table_ascii(rows)
    maybe_display_html_table(rows)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved raw results to {out_path}")


if __name__ == "__main__":
    main()
