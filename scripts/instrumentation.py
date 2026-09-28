"""Shared Trainer callback: on every logging event, prints/records peak GPU
memory and wall-clock elapsed time (plus any extra metrics already present in
the log dict, e.g. DPO's rewards/margins), and aborts loudly if peak memory
crosses a configured ceiling."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


class GPUMemoryLimitExceeded(RuntimeError):
    pass


class InstrumentationCallback(TrainerCallback):
    def __init__(self, limit_gb: float, log_path: Path, extra_keys: tuple[str, ...] = ()):
        self.limit_gb = limit_gb
        self.log_path = log_path
        self.extra_keys = extra_keys
        self.start_time: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        self.log_path.write_text("", encoding="utf-8")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return control  # skip non-step logs (e.g. the final train_runtime summary)

        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        elapsed = time.time() - self.start_time

        record = {
            "step": state.global_step,
            "loss": logs["loss"],
            "peak_mem_gb": round(peak_gb, 3),
            "elapsed_sec": round(elapsed, 1),
        }
        for key in self.extra_keys:
            if key in logs:
                record[key] = logs[key]

        extras = "  ".join(f"{k}={logs[k]:.4f}" for k in self.extra_keys if k in logs)
        print(
            f"[step {state.global_step:>5}] loss={logs['loss']:.4f}  {extras}  "
            f"peak GPU mem={peak_gb:.2f} GB  elapsed={elapsed:.1f}s"
        )
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if peak_gb > self.limit_gb:
            banner = "!" * 70
            print(banner)
            print(f"GPU MEMORY LIMIT EXCEEDED: {peak_gb:.2f} GB > {self.limit_gb:.2f} GB ceiling")
            print(banner)
            raise GPUMemoryLimitExceeded(
                f"Peak GPU memory {peak_gb:.2f} GB exceeded the {self.limit_gb:.2f} GB "
                f"ceiling at step {state.global_step}. Reduce --batch-size/--max-length "
                f"or raise --mem-limit-gb."
            )
        return control
