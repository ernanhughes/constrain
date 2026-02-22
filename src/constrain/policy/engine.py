# constrain/policy/engine.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

from constrain.data.memory import Memory
from constrain.reasoning_state import ReasoningState
from .custom_types import PolicyDecision, ThresholdProvider, Policy
from constrain.config import get_config

@dataclass
class PolicyEngine:
    policy: Policy
    threshold_provider: ThresholdProvider

    # ------------------------------------------------------------
    # Factory constructor
    # ------------------------------------------------------------

    @staticmethod
    def from_id(policy_id: int, threshold_provider: ThresholdProvider):
        from .registry import PolicyRegistry
        policy = PolicyRegistry.from_id(policy_id)
        return PolicyEngine(policy=policy, threshold_provider=threshold_provider)

    # ------------------------------------------------------------
    # Apply decision
    # ------------------------------------------------------------

    def apply(
        self,
        *,
        axes: Dict,
        flat_metrics: Dict,
        state: ReasoningState,
        memory: Memory,
        run_id: str,
        step_id: Optional[int] = None,
    ) -> PolicyDecision:

        thresholds = self.threshold_provider.get(
            cfg=get_config(),
            memory=memory,
            run_id=run_id,
            step_id=step_id,
        )

        decision = self.policy.decide(
            axes=axes,
            metrics=flat_metrics,
            state=state,
            thresholds=thresholds,
        )

        # --------------------------------------------------------
        # Optional audit logging
        # --------------------------------------------------------

        if step_id is not None:
            self._log_policy_usage(memory, run_id, step_id, thresholds, decision, self.policy.policy_id)

        return decision

    # ------------------------------------------------------------
    # Audit logging helper
    # ------------------------------------------------------------

    def _log_policy_usage(
        self,
        memory,
        run_id,
        step_id,
        thresholds,
        decision,
        policy_id,
    ):
        memory.policy_events.create_event(
            run_id=run_id,
            step_id=step_id,
            policy_id=policy_id,
            tau_soft=thresholds.tau_soft,
            tau_medium=thresholds.tau_medium,
            tau_hard=thresholds.tau_hard,
            action=decision.action,
            collapse_probability=decision.collapse_probability,
        )

    def log_policy_event(self, *, memory, run_id, step_id, decision):

        thresholds = self.threshold_provider.get(
            cfg=get_config(),
            memory=memory,
            run_id=run_id,
            step_id=step_id,
        )

        memory.policy_events.create_event(
            run_id=run_id,
            step_id=step_id,
            policy_id=self.policy.policy_id,
            tau_soft=thresholds.tau_soft,
            tau_medium=thresholds.tau_medium,
            tau_hard=thresholds.tau_hard,
            action=decision.action,
            collapse_probability=decision.collapse_probability,
        )