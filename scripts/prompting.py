"""Shared prompt template for the 311 complaint -> ticket task.

Deliberately has zero heavy dependencies (no torch/transformers) so both the
GPU training script and the CPU-only dataset generators can import it.
"""

PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Rewrite the resident complaint below into a structured 311 dispatch ticket. "
    "Respond with a single JSON object with exactly these fields: category, "
    "urgency (Low/Medium/High/Emergency), location, summary, and address "
    "(use null if no address is given).\n\n"
    "### Complaint:\n{complaint}\n\n"
    "### Ticket:\n"
)


def build_prompt(complaint: str) -> str:
    return PROMPT_TEMPLATE.format(complaint=complaint.strip())
