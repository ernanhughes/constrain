# constrain/data/schemas/causal_evaluation.py

from typing import Optional

from pydantic import BaseModel


class CausalEvaluationDTO(BaseModel):

    id: Optional[int] = None
    run_id: str

    method: str

    ate: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]

    n_samples: int
    created_at: float