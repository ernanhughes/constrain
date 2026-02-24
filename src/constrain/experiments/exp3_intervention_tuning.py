#!/usr/bin/env python3
"""
Experiment 3: Intervention Tuning

Tests different intervention strategies when collapse is detected:
- Reduce temperature (0.7 → 0.3)
- Increase temperature (0.7 → 1.2)
- Reset (clear history)
- Revert (go back 1 step)

Loads 20 conversations from DB, replays with each intervention,
measures which prevents collapse best.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from constrain.analysis.aggregation.metrics_calculator import MetricsCalculator
from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.energy.embedding.hf_embedder import HFEmbedder
from constrain.energy.embedding.sqlite_embedding_backend import \
    SQLiteEmbeddingBackend
from constrain.energy.gate import VerifiabilityGate
from constrain.energy.geometry.claim_evidence import ClaimEvidenceGeometry
from constrain.model import call_model
from constrain.reasoning_state import ReasoningState
from constrain.utils.dict_utils import flatten_numeric_dict

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ExperimentConfig:
    n_problems: int = 20
    max_depth: int = 20
    interventions: List[str] = None
    base_temperature: float = 0.7
    seed: int = 42
    
    def __post_init__(self):
        if self.interventions is None:
            self.interventions = [
                "reduce_temp",      # 0.7 → 0.3
                "increase_temp",    # 0.7 → 1.2
                "reset",            # clear history
                "revert",           # go back 1 step
            ]


@dataclass
class TrialResult:
    problem_id: int
    intervention: str
    collapse_detected_at: Optional[int]
    collapse_prevented: bool
    final_accuracy: float
    max_energy: float
    total_steps: int
    intervention_step: Optional[int]
    energy_at_intervention: Optional[float]
    accuracy_at_intervention: Optional[float]


# ============================================================
# INTERVENTION STRATEGIES
# ============================================================

class InterventionStrategy:
    """Base class for intervention strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def apply(self, state: ReasoningState, current_temp: float) -> Tuple[ReasoningState, float]:
        raise NotImplementedError
    
    def __repr__(self):
        return self.name


class ReduceTemperature(InterventionStrategy):
    def __init__(self):
        super().__init__("reduce_temp")
        self.new_temp = 0.3
    
    def apply(self, state: ReasoningState, current_temp: float) -> Tuple[ReasoningState, float]:
        state.temperature = self.new_temp
        return state, self.new_temp


class IncreaseTemperature(InterventionStrategy):
    def __init__(self):
        super().__init__("increase_temp")
        self.new_temp = 1.2
    
    def apply(self, state: ReasoningState, current_temp: float) -> Tuple[ReasoningState, float]:
        state.temperature = self.new_temp
        return state, self.new_temp


class ResetStrategy(InterventionStrategy):
    def __init__(self):
        super().__init__("reset")
    
    def apply(self, state: ReasoningState, current_temp: float) -> Tuple[ReasoningState, float]:
        state.reset()
        state.temperature = current_temp
        return state, current_temp


class RevertStrategy(InterventionStrategy):
    def __init__(self):
        super().__init__("revert")
    
    def apply(self, state: ReasoningState, current_temp: float) -> Tuple[ReasoningState, float]:
        if len(state.history) > 0:
            state.revert()
        state.temperature = current_temp
        return state, current_temp


def get_intervention_strategy(name: str) -> InterventionStrategy:
    strategies = {
        "reduce_temp": ReduceTemperature(),
        "increase_temp": IncreaseTemperature(),
        "reset": ResetStrategy(),
        "revert": RevertStrategy(),
    }
    return strategies.get(name)


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

class InterventionTuningExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.memory = Memory(get_config().db_url)
        self.cfg = get_config()
        
        # Initialize energy computation
        self.embedder = HFEmbedder(
            model_name=self.cfg.embedding_model,
            backend=SQLiteEmbeddingBackend(str(self.cfg.embedding_db)),
        )
        self.energy_computer = ClaimEvidenceGeometry(top_k=6, rank_r=4)
        self.gate = VerifiabilityGate(
            embedder=self.embedder,
            energy_computer=self.energy_computer,
        )
        
        self.results: List[TrialResult] = []
    
    def load_conversations(self, n: int = 20) -> List[Dict]:
        """Load n conversations from database."""
        logger.info(f"Loading {n} conversations from database...")
        
        # Get recent steps
        steps = self.memory.steps.get_recent_unique_steps(limit=n * self.config.max_depth)
        
        if not steps:
            raise ValueError("No conversations found in database. Run baseline first.")
        
        # Group by problem
        by_problem: Dict[int, List] = {}
        for step in steps:
            if step.problem_id not in by_problem:
                by_problem[step.problem_id] = []
            by_problem[step.problem_id].append(step)
        
        # Sort each problem by iteration
        conversations = []
        for pid, steps_list in by_problem.items():
            steps_list.sort(key=lambda s: s.iteration)
            conversations.append({
                "problem_id": pid,
                "steps": steps_list,
                "prompt": steps_list[0].prompt_text if steps_list else "",
                "gold_answer": steps_list[0].gold_answer if steps_list else "",
            })
        
        logger.info(f"Loaded {len(conversations)} conversations")
        return conversations[:n]
    
    def detect_collapse(self, energy: float) -> bool:
        """Detect if current state is collapsed."""
        return energy > self.cfg.tau_hard
    
    def replay_with_intervention(
        self,
        conversation: Dict,
        strategy: InterventionStrategy,
        intervene_at_step: Optional[int] = None,
    ) -> TrialResult:
        """Replay a conversation with a specific intervention strategy."""
        
        prompt = conversation["prompt"]
        gold_answer = conversation["gold_answer"]
        original_steps = conversation["steps"]
        
        # Initialize state
        state = ReasoningState(prompt)
        state.temperature = self.config.base_temperature
        
        collapse_detected_at = None
        collapse_prevented = False
        intervention_applied = False
        intervention_step = None
        energy_at_intervention = None
        accuracy_at_intervention = None
        
        energy_history = []
        accuracy_history = []
        
        for iteration in range(min(len(original_steps), self.config.max_depth)):
            try:
                # Generate response
                prompt_text = f"Solve step by step:\n\n{state.current}"
                reasoning = call_model(prompt_text, state.temperature)
                
                # Compute energy
                evidence_texts = self._build_evidence(prompt, state.history)
                energy_result, axes, _ = self.gate.compute_axes(
                    claim=reasoning,
                    evidence_texts=evidence_texts,
                )
                energy = axes.get("energy", 0.0)
                energy_history.append(energy)
                
                # Compute accuracy
                metrics = MetricsCalculator.compute_all(
                    reasoning=reasoning,
                    gold_answer=gold_answer,
                    energy_metrics=energy_result.to_dict(),
                    cfg=self.cfg,
                )
                accuracy = metrics.get("accuracy", 0.0)
                accuracy_history.append(accuracy)
                
                # Check for collapse
                if self.detect_collapse(energy):
                    if collapse_detected_at is None:
                        collapse_detected_at = iteration
                        
                        # Apply intervention if specified
                        if intervene_at_step is None or iteration == intervene_at_step:
                            if not intervention_applied:
                                intervention_step = iteration
                                energy_at_intervention = energy
                                accuracy_at_intervention = accuracy
                                state, state.temperature = strategy.apply(state, state.temperature)
                                intervention_applied = True
                                
                                # Check if collapse was prevented in next step
                                collapse_prevented = True
                
                # Update state (accept response)
                state.accept(reasoning)
                
            except Exception as e:
                logger.warning(f"Error at iteration {iteration}: {e}")
                break
        
        # Final accuracy
        final_accuracy = accuracy_history[-1] if accuracy_history else 0.0
        max_energy = max(energy_history) if energy_history else 0.0
        
        return TrialResult(
            problem_id=conversation["problem_id"],
            intervention=strategy.name,
            collapse_detected_at=collapse_detected_at,
            collapse_prevented=collapse_prevented if collapse_detected_at else False,
            final_accuracy=final_accuracy,
            max_energy=max_energy,
            total_steps=len(energy_history),
            intervention_step=intervention_step,
            energy_at_intervention=energy_at_intervention,
            accuracy_at_intervention=accuracy_at_intervention,
        )
    
    def _build_evidence(self, prompt: str, history: List[str]) -> List[str]:
        """Build evidence list from prompt + history."""
        from constrain.energy.utils.text_utils import split_into_sentences
        evidence = split_into_sentences(prompt)
        for past in history:
            evidence.extend(split_into_sentences(past))
        return evidence if evidence else [prompt]
    
    def run(self) -> pd.DataFrame:
        """Run the full experiment."""
        logger.info("=" * 60)
        logger.info("EXPERIMENT 3: INTERVENTION TUNING")
        logger.info("=" * 60)
        
        # Load conversations
        conversations = self.load_conversations(self.config.n_problems)
        
        # Run each intervention strategy
        for intervention_name in self.config.interventions:
            strategy = get_intervention_strategy(intervention_name)
            logger.info(f"\n🔧 Testing intervention: {strategy.name}")
            
            for conv in conversations:
                result = self.replay_with_intervention(conv, strategy)
                self.results.append(result)
                logger.info(
                    f"  Problem {conv['problem_id']}: "
                    f"collapse@{result.collapse_detected_at}, "
                    f"prevented={result.collapse_prevented}, "
                    f"final_acc={result.final_accuracy:.2f}"
                )
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Save results
        output_path = f"{self.cfg.reports_dir}/exp3_intervention_tuning.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"\n✅ Results saved to {output_path}")
        
        # Print summary
        self._print_summary(df)
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print experiment summary."""
        print("\n" + "=" * 60)
        print("INTERVENTION TUNING SUMMARY")
        print("=" * 60)
        
        # Group by intervention
        grouped = df.groupby("intervention")
        
        print("\n📊 Collapse Prevention Rate:")
        for name, group in grouped:
            n_collapse = group["collapse_detected_at"].notna().sum()
            n_prevented = group["collapse_prevented"].sum()
            rate = n_prevented / n_collapse if n_collapse > 0 else 0
            print(f"  {name:15s}: {n_prevented}/{n_collapse} ({rate:.1%})")
        
        print("\n📊 Final Accuracy (mean ± std):")
        for name, group in grouped:
            mean_acc = group["final_accuracy"].mean()
            std_acc = group["final_accuracy"].std()
            print(f"  {name:15s}: {mean_acc:.3f} ± {std_acc:.3f}")
        
        print("\n📊 Max Energy (mean ± std):")
        for name, group in grouped:
            mean_energy = group["max_energy"].mean()
            std_energy = group["max_energy"].std()
            print(f"  {name:15s}: {mean_energy:.4f} ± {std_energy:.4f}")
        
        # Best intervention
        prevention_rates = {}
        for name, group in grouped:
            n_collapse = group["collapse_detected_at"].notna().sum()
            n_prevented = group["collapse_prevented"].sum()
            prevention_rates[name] = n_prevented / n_collapse if n_collapse > 0 else 0
        
        best = max(prevention_rates, key=prevention_rates.get)
        print(f"\n🏆 Best intervention: {best} ({prevention_rates[best]:.1%} prevention)")
        
        print("=" * 60 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment 3: Intervention Tuning")
    parser.add_argument("--n-problems", type=int, default=20, help="Number of problems")
    parser.add_argument("--max-depth", type=int, default=20, help="Max conversation depth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    config = ExperimentConfig(
        n_problems=args.n_problems,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    
    experiment = InterventionTuningExperiment(config)
    experiment.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()