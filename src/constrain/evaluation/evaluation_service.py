# constrain/evaluation/evaluation_service.py

class PolicyEvaluationService:
    """
    Evaluates policy intervention effectiveness.
    
    Capabilities:
    - Reconstruct problem trajectories from step logs
    - Compute per-problem outcomes (correctness, intervention effect)
    - Compare policies with statistical significance (bootstrap)
    - Generate reports (console + CSV)
    - Query historical evaluations for comparison
    """
    
    def evaluate_run(self, run_id: str) -> PolicyEvaluationResults:
        """Reconstruct and evaluate a single run."""
        pass
    
    def compare_policies(
        self, 
        policy_ids: List[int], 
        experiment_id: Optional[int] = None
    ) -> PolicyComparisonResults:
        """Statistical comparison across policies."""
        pass
    
    def generate_report(self, run_id: str) -> str:
        """Human-readable console report."""
        pass
    
    def export_csv(self, run_ids: List[str], output_path: str):
        """Export results for external analysis."""
        pass