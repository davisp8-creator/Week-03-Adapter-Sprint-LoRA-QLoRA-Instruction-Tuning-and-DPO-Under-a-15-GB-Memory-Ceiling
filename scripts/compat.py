"""Build TrainingArguments/DPOConfig-style objects while tolerating field
drift across transformers/trl releases (e.g. transformers>=5 dropped
warmup_ratio from TrainingArguments in favor of an absolute warmup_steps)."""

from __future__ import annotations

import inspect


def build_config(cls, total_train_steps: int | None = None, **desired):
    supported = set(inspect.signature(cls.__init__).parameters)

    if (
        "warmup_ratio" in desired
        and "warmup_ratio" not in supported
        and "warmup_steps" in supported
        and total_train_steps is not None
    ):
        ratio = desired.pop("warmup_ratio")
        steps = max(1, round(ratio * total_train_steps))
        desired.setdefault("warmup_steps", steps)
        print(
            f"Note: {cls.__name__} has no warmup_ratio; using warmup_steps={steps} "
            f"({ratio:.0%} of {total_train_steps} total steps) instead."
        )

    dropped = [k for k in desired if k not in supported]
    if dropped:
        print(f"Note: {cls.__name__} doesn't accept {dropped}; skipping (using its defaults).")
        for k in dropped:
            desired.pop(k)

    return cls(**desired)
