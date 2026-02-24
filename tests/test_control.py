"""
Control system unit tests.
"""
import pytest
from constrain.control import (
    ViolationLevel,
    Action,
    classify_violation,
    compute_slope,
    update_counters,
    is_collapse,
    reset_dynamics,
    EnergyControlPolicy,
    EnergyDynamics,
)


class TestViolationClassification:
    def test_classify_none(self):
        result = classify_violation(0.2, 0.3, 0.5, 0.7)
        assert result == ViolationLevel.NONE
    
    def test_classify_soft(self):
        result = classify_violation(0.4, 0.3, 0.5, 0.7)
        assert result == ViolationLevel.SOFT
    
    def test_classify_medium(self):
        result = classify_violation(0.6, 0.3, 0.5, 0.7)
        assert result == ViolationLevel.MEDIUM
    
    def test_classify_hard(self):
        result = classify_violation(0.8, 0.3, 0.5, 0.7)
        assert result == ViolationLevel.HARD


class TestSlopeComputation:
    def test_slope_first_step(self):
        """Clarification #3: first step slope = 0"""
        result = compute_slope(0.5, None)
        assert result == 0.0
    
    def test_slope_positive(self):
        result = compute_slope(0.7, 0.5)
        assert result == 0.2
    
    def test_slope_negative(self):
        result = compute_slope(0.3, 0.5)
        assert result == -0.2


class TestCounterUpdates:
    def test_hard_counter_increments(self):
        hard, rising = update_counters(
            ViolationLevel.HARD, 0.1, 0, 0, 0.5
        )
        assert hard == 1
        assert rising == 1
    
    def test_hard_counter_resets(self):
        hard, rising = update_counters(
            ViolationLevel.NONE, 0.1, 3, 2, 0.5
        )
        assert hard == 0
    
    def test_rising_counter_first_step(self):
        """Clarification #3: first step cannot be rising"""
        hard, rising = update_counters(
            ViolationLevel.HARD, 0.1, 0, 0, None
        )
        assert rising == 0
    
    def test_rising_counter_negative_slope(self):
        hard, rising = update_counters(
            ViolationLevel.HARD, -0.1, 0, 0, 0.5
        )
        assert rising == 0


class TestCollapse:
    def test_no_collapse_single_hard(self):
        result = is_collapse(ViolationLevel.HARD, 1, 1, 3, 2)
        assert result == False
    
    def test_collapse_persistent(self):
        result = is_collapse(ViolationLevel.HARD, 3, 2, 3, 2)
        assert result == True
    
    def test_no_collapse_wrong_violation(self):
        result = is_collapse(ViolationLevel.MEDIUM, 5, 5, 3, 2)
        assert result == False


class TestPolicy:
    def test_allow_no_violation(self):
        policy = EnergyControlPolicy()
        dyn = EnergyDynamics(
            energy=0.2, prev_energy=0.2, slope=0.0,
            violation=ViolationLevel.NONE,
            consecutive_hard=0, consecutive_rising=0,
            collapse_flag=False, attempt=1, iteration=1,
        )
        decision = policy.evaluate(dyn, temperature=0.7)
        assert decision.action == Action.ALLOW
    
    def test_rollback_medium_rising(self):
        policy = EnergyControlPolicy()
        dyn = EnergyDynamics(
            energy=0.6, prev_energy=0.5, slope=0.1,
            violation=ViolationLevel.MEDIUM,
            consecutive_hard=0, consecutive_rising=1,
            collapse_flag=False, attempt=1, iteration=1,
        )
        decision = policy.evaluate(dyn, temperature=0.7)
        assert decision.action == Action.ROLLBACK
        assert decision.new_temperature < 0.7
    
    def test_reset_hard_runaway(self):
        policy = EnergyControlPolicy(runaway_slope_eps=0.05)
        dyn = EnergyDynamics(
            energy=0.9, prev_energy=0.7, slope=0.2,
            violation=ViolationLevel.HARD,
            consecutive_hard=3, consecutive_rising=2,
            collapse_flag=False, attempt=1, iteration=1,
        )
        decision = policy.evaluate(dyn, temperature=0.7)
        assert decision.action == Action.RESET


class TestResetDynamics:
    def test_reset_clears_all(self):
        prev_energy, hard, rising = reset_dynamics()
        assert prev_energy is None
        assert hard == 0
        assert rising == 0