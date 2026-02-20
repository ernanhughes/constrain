# constrain/data/schemas/step.py

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class StepDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: Optional[int] = None

    run_id: str
    problem_id: int
    iteration: int

    prompt_text: str
    reasoning_text: str

    gold_answer: Optional[str] = None
    extracted_answer: Optional[str] = None

    # --- Core Energy ---
    total_energy: float
    grounding_energy: float
    stability_energy: float

    # --- Outcomes ---
    accuracy: Optional[float] = None
    correctness: Optional[int] = None

    # --- System State ---
    temperature: float
    policy_action: str
    phase: str

    timestamp: float
