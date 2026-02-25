# constrain/services/evaluation/causal_evaluation_service.py
from __future__ import annotations

from typing import Any, Dict, Optional

from constrain.data.memory import Memory
from constrain.evaluation.causal.evaluator import CausalEvaluator, CausalEvalConfig
import json

class CausalEvaluationService:
    """
    Thin service wrapper:
      - selects runs
      - runs canonical evaluator
      - persists results
    """

    def __init__(self, memory: Memory):
        self.memory = memory
        self.evaluator = CausalEvaluator()

    def evaluate_run(self, run_id: str, cfg: Optional[CausalEvalConfig] = None) -> Optional[Dict[str, Any]]:
        result = self.evaluator.evaluate_run(memory=self.memory, run_id=run_id, cfg=cfg)
        if result is None:
            return None

        if result.get("skipped"):
            # still persist diagnostics if you want, or skip
            return result

        # Persist: keep the schema you already have
        # (adjust fields if your table expects slightly different ones)
        for method in ("naive", "ipw", "dr"):
            r = result.get(method)
            if r is None:
                continue
            self.memory.causal_evaluations.create(
                run_id=run_id,
                method=method,
                ate=r.get("ate"),
                ci_lower=r.get("ci_lower"),
                ci_upper=r.get("ci_upper"),
                n_samples=result.get("n_used"),
            )

        return result

    def evaluate_recent_runs(self, limit: int = 50) -> Dict[str, Any]:
        runs = self.memory.runs.get_recent_runs(limit=limit)
        out: Dict[str, Any] = {}
        for run in runs:
            if getattr(run, "status", None) != "completed":
                continue
            run_id = run.run_id
            out[run_id] = self.evaluate_run(run_id)
        return out
    

def main():
    memory = Memory()
    service = CausalEvaluationService(memory=memory)
    results = service.evaluate_recent_runs(limit=50)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()