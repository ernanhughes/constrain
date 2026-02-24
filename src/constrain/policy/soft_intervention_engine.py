# constrain/policy/soft_intervention_engine.py
"""
Soft Intervention Engine

Graded control modes instead of binary hard resets.
Modes: ACCEPT, NUDGE, STABILIZE, CORRECT, RESET

This replaces disruptive hard resets with stabilizing control.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from constrain.config import get_config


@dataclass
class SoftInterventionDecision:
    """Structured intervention decision with metadata."""
    mode: str  # ACCEPT, NUDGE, STABILIZE, CORRECT, RESET
    intensity: float  # 0.0 - 1.0
    new_temperature: float
    prompt_modification: Optional[str] = None
    reasoning_truncate: Optional[int] = None
    should_revert: bool = False
    should_reset: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "intensity": self.intensity,
            "new_temperature": self.new_temperature,
            "prompt_modification": self.prompt_modification,
            "reasoning_truncate": self.reasoning_truncate,
            "should_revert": self.should_revert,
            "should_reset": self.should_reset,
        }


class SoftInterventionEngine:
    """
    Maps risk scores to graded intervention modes.
    
    Control theory principle: gentle damping before hard reset.
    """
    policy_id: int = 9


    # Thresholds (tune via sweep)
    NUDGE_THRESHOLD = 0.55
    STABILIZE_THRESHOLD = 0.65
    CORRECT_THRESHOLD = 0.75
    RESET_THRESHOLD = 0.90

    # Temperature modifiers per mode
    TEMP_MODIFIERS = {
        "ACCEPT": 1.0,
        "NUDGE": 0.95,
        "STABILIZE": 0.85,
        "CORRECT": 0.70,
        "RESET": 0.50,
    }

    # Prompt injections per mode
    PROMPT_INJECTIONS = {
        "ACCEPT": None,
        "NUDGE": "Stay focused on the core question.",
        "STABILIZE": "Review your reasoning carefully. Focus on evidence.",
        "CORRECT": "Let me reconsider this step by step from the beginning.",
        "RESET": None,
    }

    def __init__(
        self,
        nudge_threshold: float = None,
        stabilize_threshold: float = None,
        correct_threshold: float = None,
        reset_threshold: float = None,
    ):
        """Allow threshold customization via config."""
        cfg = get_config()
        
        self.NUDGE_THRESHOLD = nudge_threshold or getattr(
            cfg, "soft_intervention_nudge_threshold", self.NUDGE_THRESHOLD
        )
        self.STABILIZE_THRESHOLD = stabilize_threshold or getattr(
            cfg, "soft_intervention_stabilize_threshold", self.STABILIZE_THRESHOLD
        )
        self.CORRECT_THRESHOLD = correct_threshold or getattr(
            cfg, "soft_intervention_correct_threshold", self.CORRECT_THRESHOLD
        )
        self.RESET_THRESHOLD = reset_threshold or getattr(
            cfg, "soft_intervention_reset_threshold", self.RESET_THRESHOLD
        )

    def decide(
        self,
        risk_score: float,
        current_temperature: float,
        min_temperature: float = 0.1,
    ) -> SoftInterventionDecision:
        """
        Map risk score to intervention mode and intensity.
        
        Args:
            risk_score: P(collapse | state) from learned model
            current_temperature: Current sampling temperature
            min_temperature: Floor for temperature reduction
            
        Returns:
            SoftInterventionDecision with all control parameters
        """
        # Determine mode
        if risk_score < self.NUDGE_THRESHOLD:
            mode = "ACCEPT"
        elif risk_score < self.STABILIZE_THRESHOLD:
            mode = "NUDGE"
        elif risk_score < self.CORRECT_THRESHOLD:
            mode = "STABILIZE"
        elif risk_score < self.RESET_THRESHOLD:
            mode = "CORRECT"
        else:
            mode = "RESET"

        # Compute intensity (sigmoid-like scaling within mode band)
        intensity = self._compute_intensity(risk_score, mode)

        # Compute new temperature
        temp_mod = self.TEMP_MODIFIERS[mode]
        new_temperature = max(min_temperature, current_temperature * temp_mod)

        # Get prompt modification
        prompt_mod = self.PROMPT_INJECTIONS.get(mode)

        # Compute reasoning truncation (for CORRECT mode)
        truncate = None
        if mode == "CORRECT":
            truncate = 2  # Keep last 2 turns

        # Map to legacy action format for compatibility
        should_revert = mode in ("NUDGE", "STABILIZE", "CORRECT")
        should_reset = mode == "RESET"

        return SoftInterventionDecision(
            mode=mode,
            intensity=intensity,
            new_temperature=new_temperature,
            prompt_modification=prompt_mod,
            reasoning_truncate=truncate,
            should_revert=should_revert,
            should_reset=should_reset,
        )

    def _compute_intensity(self, risk: float, mode: str) -> float:
        """Compute continuous intensity within mode band."""
        thresholds = {
            "ACCEPT": (0.0, self.NUDGE_THRESHOLD),
            "NUDGE": (self.NUDGE_THRESHOLD, self.STABILIZE_THRESHOLD),
            "STABILIZE": (self.STABILIZE_THRESHOLD, self.CORRECT_THRESHOLD),
            "CORRECT": (self.CORRECT_THRESHOLD, self.RESET_THRESHOLD),
            "RESET": (self.RESET_THRESHOLD, 1.0),
        }

        low, high = thresholds[mode]
        if high - low < 1e-6:
            return 1.0

        intensity = (risk - low) / (high - low)
        return float(np.clip(intensity, 0.0, 1.0))


class RandomizedSoftInterventionPolicy:
    """
    Wraps SoftInterventionEngine with randomized exploration.
    
    When risk > threshold:
        Randomize between recommended mode and ACCEPT
        
    This creates clean causal data for treatment effect estimation.
    """

    policy_id: int = 10

    def __init__(
        self,
        engine: SoftInterventionEngine,
        risk_threshold: float = 0.65,
        randomization_rate: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.engine = engine
        self.risk_threshold = risk_threshold
        self.randomization_rate = randomization_rate
        self.rng = np.random.RandomState(seed)

    def decide(
        self,
        risk_score: float,
        current_temperature: float,
        problem_id: str,
        iteration: int,
        run_id: str,
        min_temperature: float = 0.1,
    ) -> Tuple[SoftInterventionDecision, Dict]:
        """
        Make intervention decision with optional randomization.
        
        Args:
            risk_score: P(collapse | state)
            current_temperature: Current sampling temperature
            problem_id: For deterministic randomization
            iteration: For deterministic randomization
            run_id: For deterministic randomization
            min_temperature: Temperature floor
            
        Returns:
            decision: SoftInterventionDecision
            metadata: Decision info for logging
        """
        # Get recommended decision
        recommended = self.engine.decide(risk_score, current_temperature, min_temperature)

        # Determine if we randomize
        should_randomize = risk_score > self.risk_threshold

        if should_randomize:
            # Deterministic randomization per step (reproducible)
            seed = hash((problem_id, iteration, run_id)) % (2**31)
            step_rng = np.random.RandomState(seed)

            if step_rng.random() < self.randomization_rate:
                # Override to ACCEPT for causal comparison
                final_decision = SoftInterventionDecision(
                    mode="ACCEPT",
                    intensity=0.0,
                    new_temperature=current_temperature,
                    should_revert=False,
                    should_reset=False,
                )
                decision_type = "randomized_accept"
            else:
                # Use recommended intervention
                final_decision = recommended
                decision_type = "randomized_intervene"
        else:
            final_decision = recommended
            decision_type = "deterministic"

        metadata = {
            "risk_score": risk_score,
            "recommended_mode": recommended.mode,
            "final_mode": final_decision.mode,
            "decision_type": decision_type,
            "intensity": final_decision.intensity,
            "randomization_rate": self.randomization_rate,
            "risk_threshold": self.risk_threshold,
        }

        return final_decision, metadata

    def is_randomized_decision(self, metadata: Dict) -> bool:
        """Check if this decision was randomized (for filtering later)."""
        return metadata.get("decision_type", "").startswith("randomized")
    
