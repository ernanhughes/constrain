# constrain/experiments/recovery_experiment.py
from __future__ import annotations
import copy
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from scipy import stats

from constrain.data.memory import Memory
from constrain.data.schemas.recovery_experiment import RecoveryExperimentDTO  # Fixed: single import
from constrain.energy.gate import VerifiabilityGate
from constrain.energy.utils.text_utils import split_into_sentences
from constrain.reasoning_state import ReasoningState
from constrain.model import call_model
from constrain.config import get_config
from constrain.energy.embedding.hf_embedder import HFEmbedder
from constrain.energy.embedding.sqlite_embedding_backend import SQLiteEmbeddingBackend
from constrain.energy.geometry.claim_evidence import ClaimEvidenceGeometry

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "recovery_experiment01"
EXPERIMENT_TYPE = "intervention_recovery"  # Match one of the allowed Literal values


# ============================================================
# INTERVENTION STRATEGIES
# ============================================================

InterventionType = Literal[
    "temperature_reduction",
    "context_truncation", 
    "prompt_injection",
    "reset_to_prompt",
    "combined_stabilize"
]

@dataclass
class InterventionConfig:
    """Configuration for a single intervention strategy."""
    name: InterventionType
    description: str
    params: Dict[str, Any]
    
    @classmethod
    def temperature_reduction(cls, delta: float = 0.3, min_temp: float = 0.1):
        return cls(
            name="temperature_reduction",
            description=f"Reduce temperature by {delta} (min: {min_temp})",
            params={"delta": delta, "min_temp": min_temp}
        )
    
    @classmethod
    def context_truncation(cls, keep_last_n: int = 2):
        return cls(
            name="context_truncation",
            description=f"Keep only last {keep_last_n} reasoning steps",
            params={"keep_last_n": keep_last_n}
        )
    
    @classmethod
    def prompt_injection(cls, injection_text: str = "Review carefully. Focus on evidence."):
        return cls(
            name="prompt_injection",
            description=f"Inject guidance: '{injection_text[:50]}...'",
            params={"injection_text": injection_text}
        )
    
    @classmethod
    def reset_to_prompt(cls):
        return cls(
            name="reset_to_prompt",
            description="Reset reasoning state to original prompt",
            params={}
        )
    
    @classmethod
    def combined_stabilize(cls, temp_delta: float = 0.2, keep_n: int = 3, inject: bool = True):
        return cls(
            name="combined_stabilize",
            description="Combined: temp reduction + truncation + injection",
            params={"temp_delta": temp_delta, "keep_n": keep_n, "inject": inject}
        )


# ============================================================
# BRANCH RESULT TRACKING
# ============================================================

@dataclass
class BranchMetrics:
    """Detailed metrics for a single experimental branch."""
    survival_depth: int
    final_energy: float
    final_phase: str
    phase_transitions: List[Tuple[int, str, str]]
    energy_trajectory: List[float]
    accuracy_trajectory: List[Optional[float]]
    collapsed: bool
    collapse_iteration: Optional[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecoveryResult:
    """Complete result for one conversation branch comparison."""
    problem_id: Any
    spike_iteration: int
    spike_energy: float
    intervention_type: str
    intervention_params: Dict[str, Any]
    control_metrics: BranchMetrics
    treatment_metrics: BranchMetrics
    delta_survival: int
    delta_final_energy: float
    prevented_collapse: bool
    induced_collapse: bool
    timestamp: float
    run_id: str
    
    def to_result_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage in RecoveryExperimentDTO.per_problem_results."""
        return {
            "problem_id": str(self.problem_id),
            "spike_iteration": self.spike_iteration,
            "spike_energy": self.spike_energy,
            "intervention_type": self.intervention_type,
            "intervention_params": self.intervention_params,
            "control_metrics": self.control_metrics.to_dict(),
            "treatment_metrics": self.treatment_metrics.to_dict(),
            "delta_survival": self.delta_survival,
            "delta_final_energy": self.delta_final_energy,
            "prevented_collapse": self.prevented_collapse,
            "induced_collapse": self.induced_collapse,
            "timestamp": self.timestamp,
        }


# ============================================================
# MAIN EXPERIMENT CLASS
# ============================================================

class RecoveryExperiment:
    """
    Controlled branch experiment for intervention recovery analysis.
    """

    def __init__(
        self, 
        memory: Memory, 
        gate: VerifiabilityGate, 
        tau_hard: float,
        tau_medium: Optional[float] = None,
        tau_soft: Optional[float] = None,
    ):
        self.memory = memory
        self.gate = gate
        self.tau_hard = tau_hard
        self.tau_medium = tau_medium or (tau_hard * 0.85)
        self.tau_soft = tau_soft or (tau_hard * 0.70)
        
        self.phase_thresholds = {
            "stable": self.tau_soft,
            "drift": self.tau_medium,
            "unstable": self.tau_hard,
            "collapse": self.tau_hard * 1.5,
        }

    def run(
        self, 
        run_id: str, 
        max_depth: int = 20,
        interventions: Optional[List[InterventionConfig]] = None,
        persist_results: bool = True,
    ) -> List[RecoveryResult]:
        if interventions is None:
            interventions = [InterventionConfig.temperature_reduction()]
            
        logger.info(f"Starting recovery experiment: run={run_id}, interventions={[i.name for i in interventions]}")

        conversations = self._load_conversations(run_id)
        logger.info(f"Loaded {len(conversations)} conversations")

        all_results = []
        per_problem_results = {}  # Collect for batch persistence

        for idx, conv in enumerate(conversations):
            logger.info(f"[{idx+1}/{len(conversations)}] Problem {conv[0].problem_id}")

            for intervention in interventions:
                result = self._run_single_branch_comparison(conv, max_depth, intervention)
                if result:
                    all_results.append(result)
                    per_problem_results[str(result.problem_id)] = result.to_result_dict()
                    
                    logger.info(
                        f"  {intervention.name}: Δsurvival={result.delta_survival:+d}, "
                        f"prevented={result.prevented_collapse}, induced={result.induced_collapse}"
                    )

        if persist_results and per_problem_results:
            self._persist_experiment(run_id, per_problem_results, all_results)

        return all_results

    def _load_conversations(self, run_id: str) -> List[List]:
        steps = self.memory.steps.get_by_run(run_id)
        conversations = {}
        for step in steps:
            key = step.problem_id
            conversations.setdefault(key, []).append(step)
        for k in conversations:
            conversations[k] = sorted(conversations[k], key=lambda s: s.iteration)
        return list(conversations.values())

    def _run_single_branch_comparison(
        self, 
        steps: List, 
        max_depth: int,
        intervention: InterventionConfig,
    ) -> Optional[RecoveryResult]:
        prompt = steps[0].prompt_text
        gold_answer = getattr(steps[0], 'gold_answer', None)
        state = ReasoningState(prompt)
        
        reasoning_history = []
        energy_spike_iteration = None
        spike_energy = None
        
        for step in steps:
            state.accept(step.reasoning_text)
            reasoning_history.append(step.reasoning_text)
            if step.total_energy and step.total_energy > self.tau_hard:
                energy_spike_iteration = step.iteration
                spike_energy = step.total_energy
                break

        if energy_spike_iteration is None:
            return None

        state_control = self._clone_state_with_history(state, reasoning_history)
        state_treatment = self._clone_state_with_history(state, reasoning_history)
        
        control_metrics = self._continue_trajectory(state_control, max_depth, gold_answer, intervention=None)
        self._apply_intervention(state_treatment, intervention)
        treatment_metrics = self._continue_trajectory(state_treatment, max_depth, gold_answer, intervention=intervention)
        
        delta_survival = treatment_metrics.survival_depth - control_metrics.survival_depth
        delta_energy = treatment_metrics.final_energy - control_metrics.final_energy
        prevented = control_metrics.collapsed and not treatment_metrics.collapsed
        induced = not control_metrics.collapsed and treatment_metrics.collapsed
        
        return RecoveryResult(
            problem_id=steps[0].problem_id,
            spike_iteration=energy_spike_iteration,
            spike_energy=spike_energy or 0.0,
            intervention_type=intervention.name,
            intervention_params=intervention.params,
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            delta_survival=delta_survival,
            delta_final_energy=delta_energy,
            prevented_collapse=prevented,
            induced_collapse=induced,
            timestamp=time.time(),
            run_id=steps[0].run_id,
        )

    def _clone_state_with_history(self, state: ReasoningState, history: List[str]) -> ReasoningState:
        cloned = copy.deepcopy(state)
        cloned.history = list(history)
        return cloned

    def _continue_trajectory(
        self,
        state: ReasoningState,
        max_depth: int,
        gold_answer: Optional[str],
        intervention: Optional[InterventionConfig],
    ) -> BranchMetrics:
        energy_trajectory = []
        accuracy_trajectory = []
        phase_transitions = []
        prev_phase = self._compute_phase(state.prompt, state.history)
        
        for i in range(max_depth):
            evidence_texts = self._build_evidence(state.prompt, state.history)
            prompt_text = f"Solve step by step:\n\n{state.current}"
            
            cached = self.memory.steps.get_reasoning_by_prompt(prompt_text, state.temperature or 0.7)
            if cached:
                reasoning = cached.reasoning_text
            else:
                reasoning = call_model(prompt_text, state.temperature or 0.7)
            
            energy_result, axes, _ = self.gate.compute_axes(claim=reasoning, evidence_texts=evidence_texts)
            energy = axes.get("energy", 0.0)
            current_phase = self._compute_phase_from_energy(energy)
            
            energy_trajectory.append(energy)
            
            if gold_answer:
                acc = self._compute_accuracy(reasoning, gold_answer)
                accuracy_trajectory.append(acc)
            else:
                accuracy_trajectory.append(None)
            
            if current_phase != prev_phase:
                phase_transitions.append((i, prev_phase, current_phase))
                prev_phase = current_phase
            
            if energy > self.tau_hard:
                return BranchMetrics(
                    survival_depth=i, final_energy=energy, final_phase=current_phase,
                    phase_transitions=phase_transitions, energy_trajectory=energy_trajectory,
                    accuracy_trajectory=accuracy_trajectory, collapsed=True, collapse_iteration=i,
                )
            
            state.accept(reasoning)
        
        return BranchMetrics(
            survival_depth=max_depth,
            final_energy=energy_trajectory[-1] if energy_trajectory else 0.0,
            final_phase=prev_phase,
            phase_transitions=phase_transitions,
            energy_trajectory=energy_trajectory,
            accuracy_trajectory=accuracy_trajectory,
            collapsed=False,
            collapse_iteration=None,
        )

    def _build_evidence(self, prompt: str, history: List[str]) -> List[str]:
        evidence = split_into_sentences(prompt)
        for past in history:
            evidence.extend(split_into_sentences(past))
        return evidence if evidence else [prompt]

    def _compute_phase(self, prompt: str, history: List[str]) -> str:
        evidence = self._build_evidence(prompt, history)
        if not evidence:
            return "stable"
        _, axes, _ = self.gate.compute_axes(claim=history[-1] if history else prompt, evidence_texts=evidence)
        return self._compute_phase_from_energy(axes.get("energy", 0.0))

    def _compute_phase_from_energy(self, energy: float) -> str:
        if energy < self.tau_soft:
            return "stable"
        elif energy < self.tau_medium:
            return "drift"
        elif energy < self.tau_hard:
            return "unstable"
        else:
            return "collapse"

    def _compute_accuracy(self, reasoning: str, gold_answer: str) -> float:
        import re
        numbers = re.findall(r"-?\d+\.?\d*", reasoning)
        extracted = numbers[-1] if numbers else None
        return 1.0 if extracted == gold_answer.strip() else 0.0

    def _apply_intervention(self, state: ReasoningState, config: InterventionConfig):
        current_temp = state.temperature if state.temperature is not None else 0.7
        
        if config.name == "temperature_reduction":
            delta = config.params.get("delta", 0.3)
            min_temp = config.params.get("min_temp", 0.1)
            state.temperature = max(min_temp, current_temp - delta)
        elif config.name == "context_truncation":
            keep_n = config.params.get("keep_last_n", 2)
            if len(state.history) > keep_n:
                state.history = state.history[-keep_n:]
        elif config.name == "prompt_injection":
            injection = config.params.get("injection_text", "")
            if injection:
                state.current = f"{injection}\n\n{state.current}"
        elif config.name == "reset_to_prompt":
            state.history = []
            state.current = state.prompt
        elif config.name == "combined_stabilize":
            if config.params.get("inject", True):
                state.current = "Review carefully. Focus on evidence.\n\n" + state.current
            if config.params.get("keep_n"):
                keep_n = config.params["keep_n"]
                if len(state.history) > keep_n:
                    state.history = state.history[-keep_n:]
            if config.params.get("temp_delta"):
                delta = config.params["temp_delta"]
                state.temperature = max(0.1, (state.temperature or 0.7) - delta)
        
        logger.debug(f"Applied {config.name}: temp={state.temperature}, history_len={len(state.history)}")

    def _persist_experiment(self, run_id: str, per_problem_results: Dict[str, Dict], all_results: List[RecoveryResult]):
        """Persist results using RecoveryExperimentDTO properly."""
        try:
            # Compute summary metrics
            summary = summarize_results(all_results)
            
            # Check if experiment record already exists for this run
            existing = self.memory.recovery_experiments.get_by_run_id(run_id) if hasattr(self.memory, 'recovery_experiments') else None
            
            if existing:
                # Update existing record
                existing_results = existing.per_problem_results or {}
                existing_results.update(per_problem_results)
                
                dto = existing.model_copy(update={
                    "per_problem_results": existing_results,
                    "summary_metrics": summary,
                    "end_time": time.time(),
                    "status": "completed",
                })
                self.memory.recovery_experiments.update(existing.id, dto)
            else:
                # Create new experiment record
                dto = RecoveryExperimentDTO.create(
                    experiment_name=EXPERIMENT_NAME,
                    experiment_type=EXPERIMENT_TYPE,
                    run_ids=[run_id],
                    energy_threshold=self.tau_hard,
                    notes=f"Recovery experiment for run {run_id}",
                )
                dto = dto.add_results(
                    summary_metrics=summary,
                    per_problem_results=per_problem_results,
                ).mark_completed()
                
                if hasattr(self.memory, 'recovery_experiments'):
                    self.memory.recovery_experiments.create(dto)
                    logger.info(f"✅ Persisted recovery experiment for run {run_id}")
                else:
                    logger.warning("⚠️ Memory lacks 'recovery_experiments' store; results not persisted to DB")
                    
        except Exception as e:
            logger.warning(f"Failed to persist recovery experiment: {e}")


# ============================================================
# STATISTICAL ANALYSIS HELPERS
# ============================================================

def compute_bootstrap_ci(values: List[float], n_boot: int = 2000, ci_level: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    mean_val = np.mean(values)
    alpha = 1 - ci_level
    return mean_val, np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))


def summarize_results(results: List[RecoveryResult]) -> Dict[str, Any]:
    if not results:
        return {"error": "No results to summarize"}
    
    df = pd.DataFrame([asdict(r) for r in results])
    summaries = {}
    
    for intv_type in df["intervention_type"].unique():
        subset = df[df["intervention_type"] == intv_type]
        deltas = subset["delta_survival"].dropna().tolist()
        
        if deltas:
            mean_delta, ci_low, ci_high = compute_bootstrap_ci(deltas)
            prevented = int(subset["prevented_collapse"].sum())
            induced = int(subset["induced_collapse"].sum())
            n = len(subset)
            
            if len(deltas) >= 2:
                t_stat, p_val = stats.ttest_1samp(deltas, 0)
                significant = p_val < 0.05
            else:
                t_stat, p_val, significant = None, None, False
            
            summaries[intv_type] = {
                "n_cases": n,
                "mean_delta_survival": mean_delta,
                "ci_95": [ci_low, ci_high],
                "prevented_collapse": prevented,
                "induced_collapse": induced,
                "net_benefit": prevented - induced,
                "t_statistic": t_stat,
                "p_value": p_val,
                "significant": significant,
            }
    
    return {
        "total_cases": len(df),
        "by_intervention": summaries,
        "overall_prevention_rate": float(df["prevented_collapse"].mean()),
        "overall_harm_rate": float(df["induced_collapse"].mean()),
    }

def summarize(results):

    def extract(arm, key):
        return [r[arm][key] for r in results]

    summary = {}

    for arm in ["A", "B", "C"]:
        survival = extract(arm, "survival")
        avoided = extract(arm, "avoided_collapse")
        margins = extract(arm, "min_energy_margin")

        summary[arm] = {
            "mean_survival": np.mean(survival),
            "collapse_rate": 1 - np.mean(avoided),
            "mean_margin": np.mean(margins),
        }

    return summary

# ============================================================
# MAIN
# ============================================================

def main():
    cfg = get_config()
    memory = Memory(cfg.db_url)
    
    run_id = "run_2898a03a"
    
    embedder = HFEmbedder(model_name=cfg.embedding_model, backend=SQLiteEmbeddingBackend(str(cfg.embedding_db)))
    energy_computer = ClaimEvidenceGeometry(top_k=6, rank_r=4)
    gate = VerifiabilityGate(embedder=embedder, energy_computer=energy_computer)
    
    run_obj = memory.runs.get_by_id(run_id)
    tau_hard = getattr(run_obj, 'tau_hard_calibrated', None) or run_obj.tau_hard
    tau_medium = getattr(run_obj, 'tau_medium_calibrated', None) or run_obj.tau_medium
    tau_soft = getattr(run_obj, 'tau_soft_calibrated', None) or run_obj.tau_soft
    
    logger.info(f"Using thresholds: soft={tau_soft:.3f}, medium={tau_medium:.3f}, hard={tau_hard:.3f}")
    
    interventions = [
        InterventionConfig.temperature_reduction(delta=0.2),
        InterventionConfig.context_truncation(keep_last_n=3),
        InterventionConfig.prompt_injection(),
        InterventionConfig.combined_stabilize(),
    ]
    
    experiment = RecoveryExperiment(memory=memory, gate=gate, tau_hard=tau_hard, tau_medium=tau_medium, tau_soft=tau_soft)
    
    results = experiment.run(run_id=run_id, max_depth=20, interventions=interventions, persist_results=True)
    if not results:
        logger.warning("⚠️ No energy spikes found or all conversations survived.")
        return
    
    summarize(results)
    summary = summarize_results(results)
    
    logger.info("\n" + "=" * 70)
    logger.info("RECOVERY EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Run: {run_id}")
    logger.info(f"Total branch comparisons: {summary['total_cases']}")
    logger.info(f"Overall prevention rate: {summary['overall_prevention_rate']:.1%}")
    logger.info(f"Overall harm rate: {summary['overall_harm_rate']:.1%}")
    
    for intv_type, stats_dict in summary["by_intervention"].items():
        sig_marker = "✅" if stats_dict.get("significant") else "⚠️"
        ci_str = f"[{stats_dict['ci_95'][0]:.2f}, {stats_dict['ci_95'][1]:.2f}]"
        logger.info(f"\n{intv_type}:")
        logger.info(f"  n={stats_dict['n_cases']}, mean Δsurvival={stats_dict['mean_delta_survival']:.2f} {ci_str}")
        logger.info(f"  Prevented: {stats_dict['prevented_collapse']}, Induced: {stats_dict['induced_collapse']}")
        logger.info(f"  Net benefit: {stats_dict['net_benefit']:+d} {sig_marker} (p={stats_dict.get('p_value', 'N/A'):.3f})")
    
    logger.info("=" * 70)
    
    # Export to file
    output_path = Path(cfg.reports_dir) / f"recovery_{run_id}_{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({"run_id": run_id, "summary": summary, "results": [asdict(r) for r in results]}, f, indent=2, default=str)
    
    logger.info(f"📄 Full results exported to: {output_path}")


if __name__ == "__main__":
    main()