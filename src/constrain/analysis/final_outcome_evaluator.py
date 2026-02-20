import sqlite3
import pandas as pd
from typing import Dict, List

class FinalOutcomeEvaluator:
    """Evaluates per-problem final correctness and intervention effectiveness"""
    
    def __init__(self, db_path: str = "experiment.db"):
        self.conn = sqlite3.connect(db_path)
    
    def evaluate_run(self, run_id: str) -> Dict[str, float]:
        """
        Compute Stage 2 metrics for a single run:
        - Final accuracy per problem
        - Intervention effectiveness
        - Policy comparison readiness
        """
        # Get final step per problem (last iteration)
        final_steps = pd.read_sql_query("""
            SELECT s.problem_id, s.iteration, s.accuracy, s.correctness,
                   s.total_energy, s.policy_action
            FROM steps s
            JOIN metrics m ON s.id = m.step_id
            WHERE s.run_id = ?
            AND (s.problem_id, s.iteration) IN (
                SELECT problem_id, MAX(iteration)
                FROM steps
                WHERE run_id = ?
                GROUP BY problem_id
            )
        """, self.conn, params=(run_id, run_id))
        
        # Final accuracy (primary Stage 2 metric)
        final_accuracy = final_steps['correctness'].mean()
        
        # Intervention analysis
        interventions = pd.read_sql_query("""
            SELECT i.problem_id, i.iteration, i.threshold, i.rationale,
                   s.total_energy as energy_at_intervention
            FROM interventions i
            JOIN steps s ON i.run_id = s.run_id
            WHERE i.run_id = ?
        """, self.conn, params=(run_id,))
        
        # Effectiveness: Energy delta after intervention
        energy_deltas = []
        for _, interv in interventions.iterrows():
            # Get energy at intervention and 2 steps after
            post_steps = pd.read_sql_query("""
                SELECT total_energy FROM steps
                WHERE run_id = ? AND problem_id = ? AND iteration > ?
                ORDER BY iteration ASC LIMIT 1
            """, self.conn, params=(run_id, interv['problem_id'], interv['iteration']))
            
            if len(post_steps) > 0:
                delta = post_steps.iloc[0]['total_energy'] - interv['energy_at_intervention']
                energy_deltas.append(delta)
        
        mean_energy_delta = sum(energy_deltas) / len(energy_deltas) if energy_deltas else 0.0
        
        # Intervention rate
        intervention_rate = len(interventions) / len(final_steps) if len(final_steps) > 0 else 0.0
        
        return {
            'run_id': run_id,
            'final_accuracy': final_accuracy,
            'intervention_rate': intervention_rate,
            'mean_energy_delta_post_intervention': mean_energy_delta,
            'num_problems': len(final_steps),
            'num_interventions': len(interventions),
            'problems_with_intervention': len(interventions['problem_id'].unique())
        }
    
    def compare_policies(self, run_ids: List[str], policy_names: List[str]) -> pd.DataFrame:
        """
        Generate Stage 2 comparative table (THE CRITICAL OUTPUT)
        """
        results = []
        for run_id, policy_name in zip(run_ids, policy_names):
            eval_result = self.evaluate_run(run_id)
            eval_result['policy_name'] = policy_name
            results.append(eval_result)
        
        df = pd.DataFrame(results)
        
        # Add baseline comparison (Policy 0 = baseline)
        baseline_acc = df[df['policy_name'] == 'Baseline']['final_accuracy'].values[0]
        df['delta_accuracy_vs_baseline'] = df['final_accuracy'] - baseline_acc
        
        # Reorder columns for paper-ready output
        cols = [
            'policy_name', 'final_accuracy', 'delta_accuracy_vs_baseline',
            'intervention_rate', 'mean_energy_delta_post_intervention',
            'num_problems', 'num_interventions'
        ]
        return df[cols].sort_values('final_accuracy', ascending=False)
    
    def close(self):
        self.conn.close()