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

    total_energy: float
    grounding_energy: float
    stability_energy: float

    accuracy: Optional[float] = None
    correctness: Optional[int] = None

    temperature: float
    policy_action: str
    phase: str

    collapse_probability: Optional[float]

    intervention_mode: Optional[str] = None
    propensity_score: Optional[float] = None
    randomized: bool = False
    risk_score: Optional[float] = None
    intervention_intensity: Optional[float] = None

    timestamp: float