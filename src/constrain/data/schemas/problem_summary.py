from pydantic import BaseModel


class ProblemSummaryDTO(BaseModel):
    run_id: str
    problem_id: int
    final_correct: bool
    any_intervention: bool
    num_interventions: int
    avg_energy: float
    max_energy: float
    num_iterations: int
    intervention_helped: bool
    intervention_harmed: bool