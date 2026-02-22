from typing import Optional
from pydantic import BaseModel, ConfigDict

class PolicyEvaluationDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    run_id: str
    problem_id: int
    policy_id: int
    seed: Optional[int]
    experiment_id: Optional[int]

    final_correct: bool
    any_intervention: bool
    intervention_helped: bool
    intervention_harmed: bool

    num_iterations: int
    avg_energy: float
    max_energy: float

    tau_soft: float
    tau_medium: float
    tau_hard: float

    created_at: float