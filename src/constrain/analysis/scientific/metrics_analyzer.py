# constrain/analysis/metrics_analyzer.py
"""
Scientific-grade analysis framework for recursive stability validation.
Transforms raw logs into causal evidence with statistical rigor.
"""
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from sklearn.metrics import roc_auc_score
import logging


logger = logging.getLogger(__name__)

PHASE_MAP = {"stable": 0, "drift": 1, "unstable": 2, "collapse": 3}

class MetricsAnalyzer:
    """Scientific validation engine for recursive stability experiments"""
    
    def __init__(self):
        self.conn = sqlite3.connect("experiment.db")
    
    # ============================================================================
    # CORE ANALYSIS: Drift Quantification
    # ============================================================================
    
    def compute_per_problem_drift(self, run_id: str) -> pd.DataFrame:
        """
        Compute drift slope PER PROBLEM (not global average).
        Critical fixes:
        1. Selects 'phase' (text) from DB, converts to numeric phase_value
        2. Handles NULL accuracy values safely
        3. Uses phase_value for final state tracking
        """
        # CRITICAL: Select 'phase' (text) NOT 'phase_value' (doesn't exist in DB)
        query = """
        SELECT problem_id, iteration, total_energy, accuracy, phase
        FROM steps
        WHERE run_id = ?
        ORDER BY problem_id, iteration
        """
        df = pd.read_sql_query(query, self.conn, params=(run_id,))
        
        # ===== FIX 1: Convert phase TEXT → phase_value INTEGER =====
        df['phase_value'] = df['phase'].map(PHASE_MAP)
        
        # ===== FIX 2: Convert all columns to numeric, coerce errors to NaN =====
        df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce')
        df['total_energy'] = pd.to_numeric(df['total_energy'], errors='coerce')
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
        df['phase_value'] = pd.to_numeric(df['phase_value'], errors='coerce')  # Ensure numeric
        
        slopes = []
        for pid in df['problem_id'].unique():
            problem_df = df[df['problem_id'] == pid].sort_values('iteration').dropna(subset=['iteration'])
            
            if len(problem_df) < 3:  # Need min 3 points for slope
                continue
            
            x = problem_df['iteration'].values
            
            # ===== ENERGY SLOPE (with NaN handling) =====
            y_energy = problem_df['total_energy'].values
            mask_energy = ~np.isnan(y_energy)
            
            if np.sum(mask_energy) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x[mask_energy], y_energy[mask_energy]
                )
                energy_r2 = r_value**2
            else:
                slope, r_value, p_value, energy_r2 = np.nan, np.nan, np.nan, np.nan
            
            # ===== ACCURACY SLOPE (CRITICAL FIX) =====
            y_acc = problem_df['accuracy'].values
            mask_acc = ~np.isnan(y_acc)
            
            if np.sum(mask_acc) >= 2:
                slope_acc, _, r_acc, p_acc, _ = stats.linregress(
                    x[mask_acc], y_acc[mask_acc]
                )
                accuracy_r2 = r_acc**2
            else:
                slope_acc, r_acc, p_acc, accuracy_r2 = np.nan, np.nan, np.nan, np.nan
            
            # ===== FINAL VALUES (handle potential NaN) =====
            final_energy = problem_df['total_energy'].iloc[-1]
            final_accuracy = problem_df['accuracy'].iloc[-1]
            final_phase = problem_df['phase_value'].iloc[-1]  # NOW SAFE: phase_value exists
            
            final_energy = float(final_energy)
            final_accuracy = float(final_accuracy)
            final_phase = float(final_phase)            

            slopes.append({
                'problem_id': pid,
                'energy_slope': slope,
                'energy_r2': energy_r2,
                'energy_pval': p_value,
                'accuracy_slope': slope_acc,
                'accuracy_r2': accuracy_r2,
                'accuracy_pval': p_acc,
                'num_iterations': len(problem_df),
                'final_phase': final_phase,
                'final_energy': final_energy,
                'final_accuracy': final_accuracy
            })
        
        return pd.DataFrame(slopes)    

    def compute_drift_statistics(self, drift_df: pd.DataFrame) -> Dict[str, float]:
        """
        Aggregate per-problem drift into publication-ready statistics.
        Includes confidence intervals and significance testing.
        """
        stats_dict = {
            'mean_energy_slope': drift_df['energy_slope'].mean(),
            'std_energy_slope': drift_df['energy_slope'].std(),
            'median_energy_slope': drift_df['energy_slope'].median(),
            'positive_slope_pct': (drift_df['energy_slope'] > 0).mean() * 100,
            'significant_drift_pct': (drift_df['energy_pval'] < 0.05).mean() * 100,
            'mean_accuracy_slope': drift_df['accuracy_slope'].mean(),
            'final_energy_mean': drift_df['final_energy'].mean(),
            'final_accuracy_mean': drift_df['final_accuracy'].mean(),
            'collapse_rate': (drift_df['final_phase'] >= 2).mean() * 100  # Phase 2+ = unstable/collapse
        }
        
        # 95% confidence intervals
        n = len(drift_df)
        if n > 1:
            stats_dict['energy_slope_ci95'] = (
                stats.t.ppf(0.975, n-1) * stats_dict['std_energy_slope'] / np.sqrt(n)
            )
        
        return stats_dict
    
    # ============================================================================
    # CAUSAL ANALYSIS: Intervention Recovery Delta (IRD)
    # ============================================================================
    
    def compute_intervention_recovery(self, run_id: str) -> pd.DataFrame:
        """
        Measure CAUSAL effect of interventions: accuracy recovery after policy action.
        Critical for proving policy causes stability (not just correlates).
        """
        query = """
        SELECT problem_id, iteration, accuracy, policy_action, total_energy, phase
        FROM steps
        WHERE run_id = ?
        ORDER BY problem_id, iteration
        """
        df = pd.read_sql_query(query, self.conn, params=(run_id,))
        
        recoveries = []
        
        for pid in df['problem_id'].unique():
            problem_df = df[df['problem_id'] == pid].sort_values('iteration')
            
            # Find intervention points (non-ACCEPT actions)
            interventions = problem_df[problem_df['policy_action'] != 'ACCEPT']
            
            for _, intervention_row in interventions.iterrows():
                iter_idx = intervention_row['iteration']
                
                # Skip if near end of trajectory
                if iter_idx + 2 >= problem_df['iteration'].max():
                    continue
                
                # Accuracy BEFORE intervention (t-1)
                pre_mask = (problem_df['iteration'] == iter_idx - 1)
                if not pre_mask.any():
                    continue
                pre_acc = problem_df[pre_mask]['accuracy'].values[0]
                
                # Accuracy AFTER intervention (t+2) - allows stabilization
                post_mask = (problem_df['iteration'] == iter_idx + 2)
                if not post_mask.any():
                    continue
                post_acc = problem_df[post_mask]['accuracy'].values[0]
                
                # Baseline trajectory: same iteration in problems WITHOUT interventions
                baseline_mask = (
                    (df['problem_id'] != pid) & 
                    (df['iteration'] == iter_idx + 2) &
                    (df['policy_action'] == 'ACCEPT')
                )
                baseline_accs = df[baseline_mask]['accuracy']
                baseline_mean = baseline_accs.mean() if len(baseline_accs) > 0 else post_acc
                
                recoveries.append({
                    'problem_id': pid,
                    'intervention_step': iter_idx,
                    'policy_action': intervention_row['policy_action'],
                    'pre_accuracy': pre_acc,
                    'post_accuracy': post_acc,
                    'recovery_delta': post_acc - pre_acc,
                    'baseline_comparison': post_acc - baseline_mean,
                    'energy_at_intervention': intervention_row['total_energy'],
                    'phase_at_intervention': PHASE_MAP.get(intervention_row['phase'], -1)
                })
        
        return pd.DataFrame(recoveries)
    
    def compute_ird_statistics(self, recovery_df: pd.DataFrame) -> Dict[str, float]:
        """Aggregate intervention recovery metrics with statistical significance"""
        if len(recovery_df) == 0:
            return {'mean_recovery': 0.0, 'positive_recovery_pct': 0.0, 'n_interventions': 0}
        
        stats_dict = {
            'mean_recovery': recovery_df['recovery_delta'].mean(),
            'std_recovery': recovery_df['recovery_delta'].std(),
            'positive_recovery_pct': (recovery_df['recovery_delta'] > 0).mean() * 100,
            'mean_baseline_comparison': recovery_df['baseline_comparison'].mean(),
            'n_interventions': len(recovery_df),
            'n_problems_with_interventions': recovery_df['problem_id'].nunique()
        }
        
        # Statistical significance vs zero
        if len(recovery_df) > 1:
            t_stat, p_val = stats.ttest_1samp(recovery_df['recovery_delta'], 0)
            stats_dict['recovery_pval'] = p_val
            stats_dict['recovery_significant'] = p_val < 0.05
        
        return stats_dict
    
    # ============================================================================
    # PREDICTIVE ANALYSIS: Lead-Lag Validation
    # ============================================================================
    
    def compute_lead_lag_correlations(self, run_id: str, max_lag: int = 3) -> pd.DataFrame:
        """
        Validate predictive power: signal at t predicts collapse at t+k.
        Critical for transforming correlation into forecasting capability.
        """
        query = """
        SELECT problem_id, iteration, total_energy, grounding_energy, stability_energy,
               foreign_char_ratio, repetition_score, ascii_ratio, accuracy, phase
        FROM steps
        WHERE run_id = ?
        ORDER BY problem_id, iteration
        """
        df = pd.read_sql_query(query, self.conn, params=(run_id,))
        df['phase_value'] = df['phase'].map(PHASE_MAP)

        # Define collapse event (phase >= 2 = unstable/collapse)
        df['collapse_event'] = (df['phase_value'] >= 2).astype(int)
        
        results = []
        
        for pid in df['problem_id'].unique():
            problem_df = df[df['problem_id'] == pid].sort_values('iteration')
            
            for lag in range(1, max_lag + 1):
                if len(problem_df) <= lag:
                    continue
                
                # Signals at t, collapse at t+lag
                signals_t = problem_df.iloc[:-lag][[
                    'total_energy', 'grounding_energy', 'stability_energy',
                    'foreign_char_ratio', 'repetition_score', 'ascii_ratio'
                ]]
                collapse_t_plus_lag = problem_df['collapse_event'].iloc[lag:].values
                
                # Correlation for each signal
                for signal_col in signals_t.columns:
                    if len(signals_t) > 1 and signals_t[signal_col].std() > 1e-6:
                        corr = np.corrcoef(signals_t[signal_col], collapse_t_plus_lag)[0, 1]
                        results.append({
                            'problem_id': pid,
                            'lag': lag,
                            'signal': signal_col,
                            'corr_with_collapse': corr,
                            'abs_corr': abs(corr)
                        })
        
        return pd.DataFrame(results)
    
    def compute_predictive_auc(self, run_id: str) -> Dict[str, float]:
        """
        Compute AUC for collapse prediction using energy + foreign ratio.
        Validates if signals have genuine predictive power (not just correlation).
        """
        query = """
        SELECT problem_id, iteration, total_energy, foreign_char_ratio, phase
        FROM steps
        WHERE run_id = ?
        ORDER BY problem_id, iteration
        """
        df = pd.read_sql_query(query, self.conn, params=(run_id,))
        df['phase_value'] = df['phase'].map(PHASE_MAP)
        
        # Create features and target (collapse at next step)
        df['collapse_next'] = df.groupby('problem_id')['phase_value'].shift(-1) >= 2
        
        # Drop rows with missing targets
        df_clean = df.dropna(subset=['collapse_next'])
        
        if len(df_clean) < 10:
            return {'auc_energy': 0.5, 'auc_foreign': 0.5, 'auc_combined': 0.5}
        
        # Energy-only prediction
        try:
            auc_energy = roc_auc_score(df_clean['collapse_next'], df_clean['total_energy'])
        except Exception as e:
            logger.warning(f"Error computing energy AUC: {e}")
            auc_energy = 0.5
        
        # Foreign ratio prediction
        try:
            auc_foreign = roc_auc_score(df_clean['collapse_next'], df_clean['foreign_char_ratio'])
        except Exception as e:
            logger.warning(f"Error computing foreign AUC: {e}")
            auc_foreign = 0.5
        
        # Combined prediction (simple weighted average)
        try:
            combined_score = (
                0.6 * df_clean['total_energy'] + 
                0.4 * df_clean['foreign_char_ratio']
            )
            auc_combined = roc_auc_score(df_clean['collapse_next'], combined_score)
        except Exception as e:
            logger.warning(f"Error computing combined AUC: {e}")
            auc_combined = 0.5
        
        return {
            'auc_energy': auc_energy,
            'auc_foreign': auc_foreign,
            'auc_combined': auc_combined,
            'n_samples': len(df_clean),
            'collapse_rate': df_clean['collapse_next'].mean() * 100
        }
    
    # ============================================================================
    # PHASE TRANSITION ANALYSIS
    # ============================================================================
    
    def compute_phase_transition_matrix(self, run_id: str) -> pd.DataFrame:
        """
        Compute Markov transition matrix between phases.
        Reveals if policy changes transition probabilities (true stabilization).
        """
        query = """
        SELECT problem_id, iteration, phase
        FROM steps
        WHERE run_id = ?
        ORDER BY problem_id, iteration
        """
        df = pd.read_sql_query(query, self.conn, params=(run_id,))
        
        # Create transition pairs
        transitions = []
        for pid in df['problem_id'].unique():
            problem_df = df[df['problem_id'] == pid].sort_values('iteration')
            for i in range(len(problem_df) - 1):
                current_phase = PHASE_MAP.get(problem_df.iloc[i]['phase'], -1)
                next_phase = PHASE_MAP.get(problem_df.iloc[i + 1]['phase'], -1)
                transitions.append((current_phase, next_phase))
        
        # Build transition matrix
        phases = sorted(set([t[0] for t in transitions] + [t[1] for t in transitions]))
        matrix = pd.DataFrame(0, index=phases, columns=phases)
        
        for curr, nxt in transitions:
            matrix.loc[curr, nxt] += 1
        
        # Normalize to probabilities
        matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
        
        return matrix
    
    # ============================================================================
    # COMPREHENSIVE REPORT GENERATION
    # ============================================================================
    
    def generate_full_report(self, run_id: str, output_path: Optional[str] = None) -> Dict:
        """
        Generate publication-ready analysis report.
        Combines all analyses into single coherent narrative with statistical rigor.
        """
        logger.debug(f"Generating full analysis report for run {run_id}")
        
        # 1. Per-problem drift analysis
        drift_df = self.compute_per_problem_drift(run_id)
        drift_stats = self.compute_drift_statistics(drift_df)
        
        # 2. Intervention recovery analysis
        recovery_df = self.compute_intervention_recovery(run_id)
        recovery_stats = self.compute_ird_statistics(recovery_df)
        
        # 3. Predictive analysis
        lead_lag_df = self.compute_lead_lag_correlations(run_id, max_lag=2)
        predictive_auc = self.compute_predictive_auc(run_id)
        
        # 4. Phase transition analysis
        transition_matrix = self.compute_phase_transition_matrix(run_id)
        
        # 5. Compile comprehensive report
        report = {
            'run_id': run_id,
            'drift_analysis': {
                'per_problem_slopes': drift_df.to_dict('records'),
                'statistics': drift_stats,
                'interpretation': self._interpret_drift(drift_stats)
            },
            'intervention_analysis': {
                'recoveries': recovery_df.to_dict('records') if not recovery_df.empty else [],
                'statistics': recovery_stats,
                'interpretation': self._interpret_recovery(recovery_stats)
            },
            'predictive_analysis': {
                'lead_lag_correlations': lead_lag_df.to_dict('records') if not lead_lag_df.empty else [],
                'auc_metrics': predictive_auc,
                'interpretation': self._interpret_predictive(predictive_auc)
            },
            'phase_analysis': {
                'transition_matrix': transition_matrix.to_dict(),
                'interpretation': self._interpret_transitions(transition_matrix)
            },
            'scientific_conclusion': self._synthesize_conclusion(drift_stats, recovery_stats, predictive_auc)
        }
        
        # Save to file if requested
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.debug(f"Report saved to {output_path}")
        
        return report
    
    # ============================================================================
    # INTERPRETATION HELPERS (Scientific Narrative)
    # ============================================================================
    
    def _interpret_drift(self, stats: Dict) -> str:
        """Generate scientific interpretation of drift statistics"""
        if stats['mean_energy_slope'] > 0.02 and stats['positive_slope_pct'] > 80:
            return (
                f"Strong evidence of recursive drift: {stats['positive_slope_pct']:.1f}% of problems "
                f"show positive energy slopes (mean={stats['mean_energy_slope']:.4f}). "
                f"This confirms systematic semantic instability under uncontrolled recursion."
            )
        elif stats['mean_energy_slope'] > 0.005:
            return (
                f"Mild drift detected: mean slope {stats['mean_energy_slope']:.4f}. "
                f"Instability present but not catastrophic."
            )
        else:
            return "No significant drift detected. System remains stable under recursion."
    
    def _interpret_recovery(self, stats: Dict) -> str:
        """Generate scientific interpretation of intervention recovery"""
        if stats['n_interventions'] == 0:
            return "No interventions occurred - policy was inactive or thresholds too strict."
        
        if stats.get('recovery_significant', False) and stats['mean_recovery'] > 0.1:
            return (
                f"Strong causal evidence: interventions recover accuracy by {stats['mean_recovery']:.2%} "
                f"(p={stats.get('recovery_pval', 1.0):.3f}). Policy causes stability improvement."
            )
        elif stats['mean_recovery'] > 0.05:
            return (
                f"Moderate recovery: interventions improve accuracy by {stats['mean_recovery']:.2%}. "
                f"Suggests policy has stabilizing effect."
            )
        else:
            return (
                f"Weak or negative recovery: mean delta {stats['mean_recovery']:.2%}. "
                f"Policy may not improve downstream reasoning quality."
            )
    
    def _interpret_predictive(self, auc_metrics: Dict) -> str:
        """Generate scientific interpretation of predictive power"""
        if auc_metrics['auc_combined'] > 0.7:
            return (
                f"Strong predictive signal: combined AUC={auc_metrics['auc_combined']:.3f}. "
                f"Energy + foreign ratio predict collapse better than chance."
            )
        elif auc_metrics['auc_combined'] > 0.6:
            return f"Moderate predictive signal: AUC={auc_metrics['auc_combined']:.3f}"
        else:
            return f"Weak predictive signal: AUC={auc_metrics['auc_combined']:.3f} (near random)"
    
    def _interpret_transitions(self, matrix: pd.DataFrame) -> str:
        """Generate scientific interpretation of phase transitions"""
        # Check if policy reduces drift→collapse transitions
        if 1.0 in matrix.index and 3.0 in matrix.columns:
            drift_to_collapse = matrix.loc[1.0, 3.0] if 3.0 in matrix.columns else 0
            if drift_to_collapse < 0.2:
                return "Policy effectively blocks drift→collapse transitions (<20% probability)"
            elif drift_to_collapse < 0.4:
                return "Moderate containment of drift→collapse transitions"
            else:
                return "Weak containment: high probability of drift→collapse progression"
        return "Transition analysis incomplete (insufficient phase coverage)"
    
    def _synthesize_conclusion(self, drift_stats: Dict, recovery_stats: Dict, auc_metrics: Dict) -> str:
        """Synthesize overall scientific conclusion"""
        conclusions = []
        
        # Drift evidence
        if drift_stats['mean_energy_slope'] > 0.02:
            conclusions.append("Recursive refinement induces measurable semantic drift")
        
        # Predictive evidence
        if auc_metrics['auc_combined'] > 0.65:
            conclusions.append("Hallucination energy provides predictive instability signal")
        
        # Causal evidence
        if recovery_stats.get('recovery_significant', False) and recovery_stats['mean_recovery'] > 0.08:
            conclusions.append("Energy-thresholded interventions causally improve stability")
        
        if not conclusions:
            return (
                "Current evidence insufficient for strong claims. "
                "Recommend: 1) Increase problem count (>200), 2) Tune policy thresholds, "
                "3) Validate lead-lag relationships with larger dataset."
            )
        
        return " | ".join(conclusions)
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# ============================================================================
# CONVENIENCE FUNCTIONS FOR QUICK ANALYSIS
# ============================================================================

def quick_analysis(run_id: str, db_path: str = "experiment.db") -> None:
    """
    One-liner analysis for rapid validation during development.
    Prints key metrics to console for immediate feedback.
    """
    analyzer = MetricsAnalyzer(db_path)
    
    print("="*70)
    print(f"QUICK ANALYSIS: Run {run_id}")
    print("="*70)
    
    # Drift stats
    drift_df = analyzer.compute_per_problem_drift(run_id)
    drift_stats = analyzer.compute_drift_statistics(drift_df)
    print("\nDrift Analysis:")
    print(f"  Mean energy slope: {drift_stats['mean_energy_slope']:+.4f}")
    print(f"  Positive slopes: {drift_stats['positive_slope_pct']:.1f}%")
    print(f"  Significant drift: {drift_stats['significant_drift_pct']:.1f}%")
    
    # Intervention stats
    recovery_df = analyzer.compute_intervention_recovery(run_id)
    recovery_stats = analyzer.compute_ird_statistics(recovery_df)
    print("\nIntervention Analysis:")
    print(f"  Interventions: {recovery_stats['n_interventions']}")
    print(f"  Mean recovery: {recovery_stats['mean_recovery']:+.2%}")
    if 'recovery_pval' in recovery_stats:
        print(f"  Recovery p-value: {recovery_stats['recovery_pval']:.3f}")
    
    # Predictive stats
    auc_metrics = analyzer.compute_predictive_auc(run_id)
    print("\nPredictive Analysis:")
    print(f"  Energy AUC: {auc_metrics['auc_energy']:.3f}")
    print(f"  Foreign ratio AUC: {auc_metrics['auc_foreign']:.3f}")
    print(f"  Combined AUC: {auc_metrics['auc_combined']:.3f}")
    
    # Scientific conclusion
    report = analyzer.generate_full_report(run_id)
    print("\nScientific Conclusion:")
    print(f"  {report['scientific_conclusion']}")
    
    print("="*70)
    analyzer.close()


def generate_paper_tables(run_ids: List[str], db_path: str = "experiment.db") -> pd.DataFrame:
    """
    Generate publication-ready comparison table across multiple runs/policies.
    Outputs DataFrame suitable for LaTeX conversion.
    """
    analyzer = MetricsAnalyzer(db_path)
    results = []
    
    for run_id in run_ids:
        drift_df = analyzer.compute_per_problem_drift(run_id)
        drift_stats = analyzer.compute_drift_statistics(drift_df)
        recovery_stats = analyzer.compute_ird_statistics(
            analyzer.compute_intervention_recovery(run_id)
        )
        auc_metrics = analyzer.compute_predictive_auc(run_id)
        
        # Extract policy ID from run metadata
        cursor = analyzer.conn.cursor()
        cursor.execute("SELECT policy_id FROM runs WHERE run_id = ?", (run_id,))
        policy_id = cursor.fetchone()[0]
        
        results.append({
            'policy_id': policy_id,
            'run_id': run_id,
            'mean_energy_slope': drift_stats['mean_energy_slope'],
            'positive_slope_pct': drift_stats['positive_slope_pct'],
            'mean_recovery': recovery_stats['mean_recovery'],
            'recovery_significant': recovery_stats.get('recovery_significant', False),
            'auc_combined': auc_metrics['auc_combined'],
            'n_problems': len(drift_df),
            'collapse_rate': drift_stats['collapse_rate']
        })
    
    analyzer.close()
    return pd.DataFrame(results)