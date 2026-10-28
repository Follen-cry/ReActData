"""Default search prompts; backend/session handling is deliberately elsewhere."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptContext:
    P: dict
    state: dict
    per_round: int
    steps: int
    S: list
    top_k: int
    confirm_seeds: list
    budget_info: dict
