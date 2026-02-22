# constrain/policy/registry.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from constrain.config import get_config
from constrain.policy.policy_config import PolicyConfig
from constrain.policy.policy_params import PolicyParams
from constrain.policy.custom_types import Policy

from constrain.policy.policies import (
    BaselineAcceptPolicy,
    SimpleRevertPolicy,
    RevertCoolPolicy,
    AggressiveRevertPolicy,
    HardResetPolicy,
    MediumResetPolicy,
    RandomPolicy,
    GeometryBandPolicy,
    LearnedPolicyWrapper,
)

from constrain.policy.learned_policy import LearnedPolicy


@dataclass
class PolicyRegistry:
    config: PolicyConfig
    learned_model: Optional[object] = None
    learned_shadow: bool = False

    # ------------------------------------------------------------
    # Static Factory Entry Point
    # ------------------------------------------------------------

    @staticmethod
    def from_id(policy_id: int) -> Policy:

        cfg = get_config()

        # -----------------------------
        # Build PolicyConfig
        # -----------------------------

        policy_config = PolicyConfig(
            min_temperature=cfg.min_temperature,
            revert_cooldown_factor=cfg.revert_cooldown_factor,
            aggressive_cooldown_factor=cfg.aggressive_cooldown_factor,
            reset_cooldown_factor=cfg.reset_cooldown_factor,
            random_revert_probability=cfg.random_revert_probability,
            geometry_pr_threshold=cfg.geometry_pr_threshold,
            geometry_sensitivity_threshold=cfg.geometry_sensitivity_threshold,
            geometry_band_width_factor=cfg.geometry_band_width_factor,
        )

        # -----------------------------
        # Load learned model if needed
        # -----------------------------

        learned_model = None
        if policy_id == 99:

            if not hasattr(cfg, "learned_model_path"):
                raise ValueError("learned_model_path missing from config.")

            learned_model = LearnedPolicy(
                model_path=cfg.learned_model_path,
                threshold=cfg.learned_policy_threshold,
            )

        registry = PolicyRegistry(
            config=policy_config,
            learned_model=learned_model,
            learned_shadow=cfg.learned_policy_shadow,
        )

        return registry.build(policy_id)

    # ------------------------------------------------------------
    # Internal Builder
    # ------------------------------------------------------------

    def build(self, policy_id: int) -> Policy:

        params = PolicyParams(
            min_temperature=self.config.min_temperature,
            revert_cooldown_factor=self.config.revert_cooldown_factor,
            aggressive_cooldown_factor=self.config.aggressive_cooldown_factor,
            reset_cooldown_factor=self.config.reset_cooldown_factor,
        )

        if policy_id == 0:
            return BaselineAcceptPolicy()

        if policy_id == 1:
            return SimpleRevertPolicy(params=params)

        if policy_id == 2:
            return RevertCoolPolicy(params=params)

        if policy_id == 3:
            return AggressiveRevertPolicy(params=params)

        if policy_id == 4:
            return HardResetPolicy(params=params)

        if policy_id == 5:
            return MediumResetPolicy(params=params)

        if policy_id == 6:
            return RandomPolicy(
                params=params,
                revert_probability=self.config.random_revert_probability,
            )

        if policy_id == 8:
            return GeometryBandPolicy(
                params=params,
                pr_threshold=self.config.geometry_pr_threshold,
                sensitivity_threshold=self.config.geometry_sensitivity_threshold,
                band_width_factor=self.config.geometry_band_width_factor,
            )

        if policy_id == 99:
            if self.learned_model is None:
                raise ValueError("Learned policy selected but model not loaded.")

            return LearnedPolicyWrapper(
                learned_model=self.learned_model,
                shadow=self.learned_shadow,
            )

        raise ValueError(f"Unknown policy_id: {policy_id}")