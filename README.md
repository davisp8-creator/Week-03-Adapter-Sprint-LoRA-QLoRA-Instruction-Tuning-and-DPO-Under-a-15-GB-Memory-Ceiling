# CivicDesk 311 Complaint Rewriter

Small NLP assistant that rewrites messy, free-text resident 311 complaints
into structured dispatch tickets (`category`, `urgency`, `location`,
`summary`, `address`), built to fit inside a Google Colab free-tier T4 GPU's
~15 GB memory ceiling. See [ASSIGNMENT.md](ASSIGNMENT.md) for the original
project brief.

Pipeline: synthesize an instruction dataset -> QLoRA fine-tune a sub-1B base
model -> synthesize a targeted DPO preference set -> DPO-align the adapter on
top of the frozen SFT policy -> evaluate base vs. SFT vs. DPO on held-out
complaints. Every training run logs peak GPU memory and wall-clock time; see
[RUNLOG.md](RUNLOG.md) for the real failures hit along the way, how each was
diagnosed, and how it was fixed.

## Repo layout

| Path | Purpose |
|---|---|
| `env_setup.py` | Detects Windows VM vs. Colab, installs the matching torch build (CUDA vs. CPU) plus the shared HF/PEFT/TRL stack |
| `scripts/generate_synthetic_complaints.py` | Synthesizes the 400-example SFT instruction dataset |
| `scripts/generate_dpo_pairs.py` | Synthesizes the 150-pair DPO preference dataset (address-hallucination axis) |
| `scripts/prompting.py` | Shared prompt template used by every stage |
| `scripts/compat.py` | Tolerates `transformers`/`trl` API drift across releases (see RUNLOG.md) |
| `scripts/instrumentation.py` | Shared Trainer callback: logs peak GPU memory + wall-clock time every N steps, aborts loudly over the memory ceiling |
| `scripts/train_qlora.py` | 4-bit NF4 QLoRA SFT fine-tune |
| `scripts/train_dpo.py` | DPO pass (trl) on top of the saved SFT adapter, frozen SFT policy as reference |
| `scripts/evaluate_models.py` | Side-by-side base vs. SFT vs. DPO comparison on held-out complaints |
| `scripts/run_pipeline.py` | Controller: runs every stage above end to end on either environment, then commits results + updates this README back to GitHub |
| `data/` | Generated datasets (checked in for reproducibility) |
| `outputs/` | Trained adapters + per-run memory/time logs (checked in by the controller) |
| `runs/` | One JSON record per controller run (system, timings, peak memory) |

## Model

Base model: [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2)
(124M params, well under the sub-1B budget). It's small enough that 4-bit
quantization isn't strictly *necessary* for memory -- it's used anyway
because the point of the exercise is practicing the QLoRA workflow under a
memory ceiling, and the same scripts work unchanged on a larger sub-1B model
that actually needed it.

## Setup

Works on either a Windows Server VM (CPU-only, Intel) or a Google Colab T4
GPU notebook -- `env_setup.py` detects which one it's on and installs
accordingly.

```bash
git clone https://github.com/davisp8-creator/Week-03-Adapter-Sprint-LoRA-QLoRA-Instruction-Tuning-and-DPO-Under-a-15-GB-Memory-Ceiling.git
cd Week-03-Adapter-Sprint-LoRA-QLoRA-Instruction-Tuning-and-DPO-Under-a-15-GB-Memory-Ceiling
python env_setup.py
```

### GitHub push access (optional, only needed for `run_pipeline.py`)

The controller script can commit its results (adapters, logs, this README's
Run History table) back to this repo. It needs a **fine-grained GitHub
personal access token**, scoped to *only this repository*, with **Contents:
Read and write** permission -- create one at GitHub -> Settings -> Developer
settings -> Fine-grained tokens.

Never paste the token directly into a notebook cell that gets saved, and
never commit it. On Colab, use the built-in Secrets manager (key icon in the
left sidebar):

```python
from google.colab import userdata
import os
os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
```

On the Windows VM, set it as a real environment variable for the session
(PowerShell: `$env:GITHUB_TOKEN = "..."`). If `GITHUB_TOKEN` isn't set,
`run_pipeline.py` still runs every stage and writes results locally -- it
just skips the push and says so.

## Running the full pipeline

```bash
python scripts/run_pipeline.py
```

This runs, in order: environment detection/setup, SFT dataset generation,
DPO dataset generation, QLoRA SFT training (GPU only), DPO training (GPU
only, needs the SFT adapter), evaluation (GPU if available, else CPU, needs
both adapters) -- then writes a `runs/<timestamp>.json` record, updates this
README's Run History table, and commits + pushes the results.

Stages that need a GPU are skipped (not failed) with a clear reason when run
on the CPU-only Windows VM; a later run on Colab, or a `git pull` of adapters
someone already trained, picks up where it left off.

Flags: `--skip-sft`, `--skip-dpo`, `--skip-eval`, `--no-commit` (run
everything, touch no git state at all), `--no-push` (commit locally, don't
push).

## Running stages individually

```bash
python scripts/generate_synthetic_complaints.py --n 400 --out data/311_complaints.jsonl
python scripts/generate_dpo_pairs.py --n 150 --out data/dpo_address_preferences.jsonl
python scripts/train_qlora.py --data data/311_complaints.jsonl --output-dir outputs/qlora-gpt2-311
python scripts/train_dpo.py --sft-adapter outputs/qlora-gpt2-311/adapter --output-dir outputs/dpo-gpt2-311
python scripts/evaluate_models.py --sft-adapter outputs/qlora-gpt2-311/adapter --dpo-adapter outputs/dpo-gpt2-311/adapter
```

## Environment Comparison

The same `scripts/run_pipeline.py` controller was run unmodified on both
target environments. It correctly ran the CPU-safe stages and skipped the
GPU-only stages (with a stated reason, not a crash) where there's no CUDA
GPU:

| | Windows VM (CPU-only, Intel) | Google Colab (T4 GPU) |
|---|---|---|
| Ran | data generation only | QLoRA SFT training |
| Skipped | `train_qlora`, `train_dpo`, `evaluate` -- all "no CUDA GPU available" (eval also "adapters not both present") | -- |
| Wall clock | 2s | 1m 23s |
| Peak GPU memory | N/A (no GPU) | 0.76 GB (14 GB ceiling) |

This is the intended behavior, not a limitation being worked around: the
Windows VM is meant to handle the CPU-safe half of the pipeline (data
synthesis, and evaluation once adapters exist locally), while GPU-bound
training only ever runs on Colab. Once a Colab run produces the SFT and DPO
adapters and they're committed back to the repo (see Run History below), a
later `run_pipeline.py` run on the Windows VM will also execute `evaluate`
against them on CPU.

## Run History

Updated automatically by `scripts/run_pipeline.py` on every run (newest
first). Rows above the markers below were entered manually, before the
controller script existed.

<!-- RUN_HISTORY:START -->
| Date | System | GPU | Steps run | Total time | Peak GPU mem |
|---|---|---|---|---|---|
| 2026-09-28 | windows (Windows) | CPU only | gen_sft_data, gen_dpo_data | 2s | N/A (no GPU) |
| 2026-09-27 | Google Colab (manual run, pre-controller) | T4 | sft | 1m 23s | 0.76 GB |
<!-- RUN_HISTORY:END -->
