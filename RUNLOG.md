# Run Log: Failures, Root Causes, Fixes

This is the diagnostic log for the CivicDesk pipeline: real failures hit
during development, how each was traced to its actual cause, and how it was
fixed. For the running ledger of per-run timing/memory numbers, see the
"Run History" table in [README.md](README.md) (updated automatically by
`scripts/run_pipeline.py`).

## Failure 1: Colab kernel launcher rejects our CLI args

**Symptom** (running any of our scripts inside a Colab cell):
```
usage: colab_kernel_launcher.py [-h] [--n N] [--out OUT] [--seed SEED] ...
colab_kernel_launcher.py: error: unrecognized arguments: -f /root/.local/share/jupyter/runtime/kernel-68bb7236-....json
SystemExit: 2
```

**Root cause**: Jupyter/Colab injects its own `-f <kernel.json>` argument into
`sys.argv` on every cell execution (it's how the kernel finds its connection
file). `argparse.parse_args()` rejects any argument it doesn't recognize, so
every one of our scripts crashed immediately when run as `!python script.py`
inside a notebook, even though the exact same command worked fine from a
plain shell.

**Fix**: every script's entrypoint uses `parser.parse_known_args()` instead
of `parser.parse_args()`, so unrecognized flags (Colab's `-f`, or anything
else) are silently ignored instead of raising `SystemExit`. Applied to
`generate_synthetic_complaints.py`, `generate_dpo_pairs.py`, `train_qlora.py`,
`train_dpo.py`, and `evaluate_models.py`.

**Verified**: re-ran `generate_synthetic_complaints.py` locally with a
simulated `-f /root/.../kernel-fake.json` argument appended; it completed
normally and produced the same output. Later confirmed on the real Colab
session as well.

## Failure 2: `transformers>=5` dropped `TrainingArguments(warmup_ratio=...)`

**Symptom** (first real QLoRA training attempt on Colab, `transformers==5.16.1`):
```
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'
```

**Root cause**: `warmup_ratio` is a long-standing, very commonly used
`TrainingArguments` field, so this wasn't an obvious version mismatch at
first glance. Rather than guess, the installed package was introspected
directly:
```python
import inspect
from transformers import TrainingArguments
'warmup_ratio' in inspect.signature(TrainingArguments.__init__).parameters  # -> False
'warmup_steps' in inspect.signature(TrainingArguments.__init__).parameters  # -> True
```
This confirmed `transformers>=5` removed the ratio-based warmup field
entirely, keeping only the absolute-step-count `warmup_steps`. The same
signature check against `trl`'s `DPOConfig` (which subclasses the same base)
showed the identical gap.

**Fix**: added `scripts/compat.py::build_config()`. It introspects the
installed class's `__init__` signature at call time; if `warmup_ratio` isn't
supported but `warmup_steps` is, it converts the ratio into an absolute step
count using the pipeline's own computed total-training-steps figure (so the
3%-of-training warmup behavior is preserved, not just dropped). Any other
kwarg the installed version doesn't recognize is dropped with a printed note
rather than crashing. Used by both `train_qlora.py` (`TrainingArguments`) and
`train_dpo.py` (`DPOConfig`), so the pipeline tolerates future field drift in
either library without another hard crash.

**Verified**: re-ran the same training command on the same Colab session
after the fix. Training completed end to end: 75 steps, loss 3.173 -> 1.577,
peak GPU memory 0.76 GB, wall clock 83.1 s. See README.md Run History.

## Failure 3: Colab's preinstalled `torchao` breaks every `PeftModel` load

**Symptom** (running `evaluate_models.py` on Colab, right after `train_qlora`
and `train_dpo` had both already completed successfully in the same
pipeline run):
```
File ".../peft/tuners/lora/torchao.py", line 160, in dispatch_torchao
    if not is_torchao_available():
File ".../peft/import_utils.py", line 147, in is_torchao_available
    raise ImportError(
ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported
```

**Root cause**: read `peft`'s actual source
(`peft/import_utils.py::is_torchao_available`) rather than guessing. It
doesn't just return `False` when `torchao` is missing or unusable -- if
`torchao` **is** importable but below version 0.16.0, it deliberately
*raises* `ImportError`. Colab's default image ships `torchao==0.10.0`
preinstalled (for its own torch build), which is unrelated to anything our
pipeline uses -- we quantize with bitsandbytes 4-bit NF4, not torchao. Every
`PeftModel.from_pretrained` / `get_peft_model` call walks all of `peft`'s
tuner dispatchers, including the torchao one, so this broke adapter loading
entirely, not just a torchao-specific code path.

**Fix**: `env_setup.py::_remove_incompatible_torchao()` checks the installed
`torchao` version after installing our stack and uninstalls it if it's below
0.16.0 (leaves it alone if absent or already compatible). Since we never use
torchao, removing it is safer than upgrading a package we don't otherwise
touch.

**Verified so far**: confirmed via `peft`'s actual installed source (above)
that this is the real mechanism, and confirmed locally that
`_remove_incompatible_torchao()` is a safe no-op on a box without torchao
installed (the Windows VM). Not yet re-run on Colab against a real
`torchao==0.10.0` install -- that's the next real-environment check.

## Known unexercised path: the GPU memory ceiling has never actually been hit

`scripts/instrumentation.py::InstrumentationCallback` aborts training loudly
(raises `GPUMemoryLimitExceeded`) if peak GPU memory crosses `--mem-limit-gb`
(default 14.0 GB). In every real run so far, peak usage has been ~0.76 GB --
`openai-community/gpt2` (124M params) is small enough that this ceiling is
nowhere close, even without 4-bit quantization. The 4-bit NF4 QLoRA setup was
still built and used deliberately, to practice and prove out the workflow
that would matter on a larger sub-1B model, but the fail-loud path itself is
therefore unverified against a real OOM.

To actually exercise it: temporarily pass `--mem-limit-gb 0.5` (well below
the observed ~0.76 GB peak) and confirm training aborts with the loud
`!!!` banner and a `GPUMemoryLimitExceeded` traceback, then remove the flag.
