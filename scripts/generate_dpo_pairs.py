"""
Build a DPO preference dataset targeting one specific failure mode: address
hallucination when a resident's complaint never gave a street address.

For each of --n synthetic no-address complaints:
  - chosen   -> ticket keeps address: null, and the summary explicitly tells
               dispatch to escalate back to the resident for the exact location.
  - rejected -> ticket invents a plausible-looking street address that was
               never mentioned in the complaint, and states it confidently.

Category, urgency, and the general location description are kept identical
between chosen and rejected so DPO training penalizes the address fabrication
specifically, not some other confound.

Output JSONL, one record per line, matching the standard DPOTrainer (trl)
schema:
    {"prompt": "...", "chosen": "<json ticket>", "rejected": "<json ticket>"}

Usage:
    python scripts/generate_dpo_pairs.py --n 150 --out data/dpo_address_preferences.jsonl
    python scripts/generate_dpo_pairs.py --preview 3   # print samples, no file
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from generate_synthetic_complaints import (
    BODY_TEMPLATES,
    CLOSERS,
    DURATIONS,
    EXTRAS,
    INCIDENTS,
    LANDMARKS,
    NEIGHBORHOODS,
    OPENERS,
    STREET_NAMES,
    STREET_SUFFIXES,
    inject_typos,
    random_address,
)
from prompting import build_prompt

ESCALATION_NOTE = (
    "No address was given -- escalate to the resident to confirm the exact "
    "location before dispatch."
)


def render_complaint(rng: random.Random, detail: str, location_mention: str) -> str:
    opener = rng.choice(OPENERS)
    body = rng.choice(BODY_TEMPLATES).format(
        detail=detail,
        location_mention=location_mention,
        extra=rng.choice(EXTRAS),
        duration=rng.choice(DURATIONS),
    )
    closer = rng.choice(CLOSERS)
    raw = " ".join(part for part in (opener, body, closer) if part).strip()
    raw = re.sub(r"\s+", " ", raw)
    typo_rate = rng.choices([0.0, 0.02, 0.05, 0.09], weights=[0.15, 0.4, 0.3, 0.15])[0]
    return inject_typos(raw, typo_rate, rng)


def build_pair(rng: random.Random) -> dict:
    category = rng.choice(list(INCIDENTS))
    detail, urgency, base_summary = rng.choice(INCIDENTS[category])

    neighborhood = rng.choice(NEIGHBORHOODS)
    landmark = rng.choice(LANDMARKS)
    cross = rng.choice(STREET_NAMES)

    # The raw complaint text never states a real address -- same no-address
    # location cues as generate_synthetic_complaints.py's has_address=False path.
    location_mention = rng.choice(
        [
            f" near {landmark}",
            f" over by {landmark}",
            f" in the {neighborhood} area",
            "",
        ]
    )
    complaint = render_complaint(rng, detail, location_mention)

    true_location = rng.choice(
        [
            f"Near {landmark}, {neighborhood}",
            f"{neighborhood} (near intersection of {cross} {rng.choice(STREET_SUFFIXES)})",
            f"General vicinity of {landmark}",
        ]
    )

    chosen_ticket = {
        "category": category,
        "urgency": urgency,
        "location": true_location,
        "summary": f"{base_summary} near {landmark}. {ESCALATION_NOTE}",
        "address": None,
    }

    fake_address = random_address(rng)
    rejected_ticket = {
        "category": category,
        "urgency": urgency,
        "location": f"Near {fake_address}, {neighborhood}",
        "summary": f"{base_summary} at {fake_address}.",
        "address": fake_address,
    }

    return {
        "prompt": build_prompt(complaint),
        "chosen": json.dumps(chosen_ticket, ensure_ascii=False),
        "rejected": json.dumps(rejected_ticket, ensure_ascii=False),
    }


def generate_dataset(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [build_pair(rng) for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=150, help="number of preference pairs to generate")
    parser.add_argument("--out", type=str, default="data/dpo_address_preferences.jsonl")
    parser.add_argument("--seed", type=int, default=123, help="distinct seed from the SFT generator, for fresh scenarios")
    parser.add_argument("--preview", type=int, default=0, help="print N sample pairs and exit without writing")
    # parse_known_args so this also runs unmodified inside Jupyter/Colab.
    args, _unknown = parser.parse_known_args()

    records = generate_dataset(args.n, args.seed)

    if args.preview:
        for r in records[: args.preview]:
            print(json.dumps(r, indent=2))
            print("-" * 60)
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} preference pairs to {out_path}")


if __name__ == "__main__":
    main()
