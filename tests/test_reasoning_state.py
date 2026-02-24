# tests/test_reasoning_state.py

"""
Comprehensive test suite for ReasoningState stack-based state machine.

Tests validate:
1. True push/pop semantics (revert actually removes bad states)
2. Temperature tracked per step
3. Cannot pop below initial prompt
4. History integrity after operations
5. Reset functionality
6. Branching support (if implemented)
7. Metrics preservation
8. Temperature setter support
"""

import pytest
from unittest.mock import patch
from constrain.reasoning_state import ReasoningState


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def base_state():
    """Create a basic ReasoningState for testing."""
    return ReasoningState(prompt="Solve this math problem", temperature=0.7)


@pytest.fixture
def mock_config():
    """Mock configuration to avoid loading real config."""
    with patch('constrain.reasoning_state.get_config') as mock_cfg:
        mock_cfg.return_value.initial_temperature = 0.7
        yield mock_cfg


# ─────────────────────────────────────────────────────────────
# INITIALIZATION TESTS
# ─────────────────────────────────────────────────────────────

class TestInitialization:
    """Test ReasoningState initialization."""
    
    def test_init_with_default_temperature(self, mock_config):
        """State initializes with config default temperature."""
        state = ReasoningState(prompt="test prompt")
        
        assert state.prompt == "test prompt"
        assert state.temperature == 0.7
        assert state.depth == 1  # Only initial prompt
        assert len(state.history) == 1
    
    def test_init_with_explicit_temperature(self, mock_config):
        """State initializes with provided temperature."""
        state = ReasoningState(prompt="test prompt", temperature=0.5)
        
        assert state.temperature == 0.5
    
    def test_initial_history_contains_only_prompt(self, mock_config):
        """Initial history contains only the prompt."""
        state = ReasoningState(prompt="initial prompt")
        
        assert state.history == ["initial prompt"]
        assert state.current == "initial prompt"
    
    def test_initial_depth_is_one(self, mock_config):
        """Initial stack depth is 1 (prompt only)."""
        state = ReasoningState(prompt="test")
        
        assert state.depth == 1


# ─────────────────────────────────────────────────────────────
# PUSH/ACCEPT TESTS
# ─────────────────────────────────────────────────────────────

class TestPushAccept:
    """Test push and accept operations."""
    
    def test_push_adds_to_stack(self, base_state):
        """Push adds new state to stack."""
        base_state.push("step 1 reasoning", temperature=0.6)
        
        assert base_state.depth == 2
        assert base_state.current == "step 1 reasoning"
        assert base_state.temperature == 0.6
    
    def test_accept_is_alias_for_push(self, base_state):
        """Accept is backward-compatible alias for push."""
        base_state.accept("step 1 reasoning", temperature=0.6)
        
        assert base_state.depth == 2
        assert base_state.current == "step 1 reasoning"
    
    def test_push_inherits_temperature_if_not_provided(self, base_state):
        """Push inherits current temperature if not specified."""
        base_state.push("step 1")  # No temperature specified
        
        assert base_state.temperature == 0.7  # Inherited from initial
    
    def test_multiple_pushes_accumulate(self, base_state):
        """Multiple pushes accumulate on stack."""
        base_state.push("step 1", temperature=0.6)
        base_state.push("step 2", temperature=0.5)
        base_state.push("step 3", temperature=0.4)
        
        assert base_state.depth == 4  # prompt + 3 steps
        assert base_state.current == "step 3"
        assert base_state.temperature == 0.4
    
    def test_history_reflects_all_states(self, base_state):
        """History reflects all states in order."""
        base_state.push("step 1")
        base_state.push("step 2")
        
        assert len(base_state.history) == 3
        assert base_state.history[0] == "Solve this math problem"
        assert base_state.history[1] == "step 1"
        assert base_state.history[2] == "step 2"


# ─────────────────────────────────────────────────────────────
# POP/REVERT TESTS (CRITICAL FIX)
# ─────────────────────────────────────────────────────────────

class TestPopRevert:
    """Test pop and revert operations - THE CRITICAL FIX."""
    
    def test_revert_actually_pops_from_stack(self, base_state):
        """
        CRITICAL: revert() must actually remove bad states from stack.
        This is the fix for runaway hallucination recovery.
        """
        base_state.push("step 1")
        base_state.push("step 2_hallucinated")
        
        assert base_state.depth == 3
        assert base_state.current == "step 2_hallucinated"
        
        # CRITICAL FIX: revert must actually pop
        base_state.revert()
        
        assert base_state.depth == 2  # step_2_hallucinated REMOVED
        assert base_state.current == "step 1"
        assert "step_2_hallucinated" not in base_state.history
    
    def test_pop_returns_true_on_success(self, base_state):
        """Pop returns True when successful."""
        base_state.push("step 1")
        
        result = base_state.pop()
        
        assert result == True
    
    def test_pop_returns_false_when_only_prompt_remains(self, base_state):
        """Pop returns False when cannot pop further."""
        result = base_state.pop()  # Try to pop prompt
        
        assert result == False
        assert base_state.depth == 1  # Still has prompt
    
    def test_pop_multiple_steps(self, base_state):
        """Pop can remove multiple steps at once."""
        base_state.push("step 1")
        base_state.push("step 2")
        base_state.push("step 3")
        
        assert base_state.depth == 4
        
        base_state.pop(steps=2)
        
        assert base_state.depth == 2  # prompt + step 1
        assert base_state.current == "step 1"
    
    def test_pop_cannot_exceed_stack_depth(self, base_state):
        """Pop cannot remove more steps than exist."""
        base_state.push("step 1")
        
        # Try to pop 10 steps (only 1 exists)
        base_state.pop(steps=10)
        
        assert base_state.depth == 1  # Only prompt remains
    
    def test_temperature_restored_after_revert(self, base_state):
        """Temperature is restored to previous step's value after revert."""
        base_state.push("step 1", temperature=0.6)
        base_state.push("step 2", temperature=0.3)
        
        assert base_state.temperature == 0.3
        
        base_state.revert()
        
        assert base_state.temperature == 0.6  # Restored to step 1's temp
    
    def test_revert_is_alias_for_pop(self, base_state):
        """Revert is backward-compatible alias for pop."""
        base_state.push("step 1")
        base_state.push("step 2")
        
        base_state.revert()
        
        assert base_state.depth == 2
        assert base_state.current == "step 1"
    
    def test_history_integrity_after_revert(self, base_state):
        """History remains consistent after revert operations."""
        base_state.push("step 1")
        base_state.push("step 2")
        base_state.push("step 3")
        
        base_state.revert()
        
        # Verify history is clean
        assert len(base_state.history) == 3  # prompt + step 1 + step 2
        assert base_state.history[-1] == "step 2"
        assert "step 3" not in base_state.history


# ─────────────────────────────────────────────────────────────
# TEMPERATURE TESTS
# ─────────────────────────────────────────────────────────────

class TestTemperature:
    """Test temperature tracking and modification."""
    
    def test_temperature_setter_updates_current_step(self, base_state):
        """Temperature setter updates current stack top."""
        base_state.temperature = 0.5
        
        assert base_state.temperature == 0.5
    
    def test_temperature_tracked_per_step(self, base_state):
        """Each step maintains its own temperature."""
        base_state.push("step 1", temperature=0.6)
        base_state.push("step 2", temperature=0.4)
        base_state.push("step 3", temperature=0.2)
        
        # Verify each temperature is preserved
        base_state.revert()  # Back to step 2
        assert base_state.temperature == 0.4
        
        base_state.revert()  # Back to step 1
        assert base_state.temperature == 0.6
    
    def test_temperature_persists_after_revert(self, base_state):
        """Temperature changes persist through revert operations."""
        base_state.push("step 1", temperature=0.6)
        base_state.temperature = 0.5  # Modify step 1's temp
        base_state.push("step 2", temperature=0.4)
        
        base_state.revert()
        
        assert base_state.temperature == 0.5  # Modified temp preserved


# ─────────────────────────────────────────────────────────────
# RESET TESTS
# ─────────────────────────────────────────────────────────────

class TestReset:
    """Test reset functionality."""
    
    def test_reset_clears_history(self, base_state):
        """Reset clears all reasoning history."""
        base_state.push("step 1")
        base_state.push("step 2")
        base_state.push("step 3")
        
        base_state.reset()
        
        assert base_state.depth == 1
        assert len(base_state.history) == 1
        assert base_state.history[0] == "Solve this math problem"
    
    def test_reset_restores_initial_temperature(self, base_state):
        """Reset restores initial temperature."""
        base_state.push("step 1", temperature=0.3)
        base_state.temperature = 0.2
        
        base_state.reset()
        
        assert base_state.temperature == 0.7  # Initial temp
    
    def test_reset_preserves_prompt(self, base_state):
        """Reset preserves the original prompt."""
        base_state.push("step 1")
        
        base_state.reset()
        
        assert base_state.prompt == "Solve this math problem"
        assert base_state.current == "Solve this math problem"


# ─────────────────────────────────────────────────────────────
# BRANCHING TESTS (OPTIONAL)
# ─────────────────────────────────────────────────────────────

class TestBranching:
    """Test branching functionality for alternative paths."""
    
    def test_branch_from_creates_independent_state(self, base_state):
        """Branch creates independent state from specified depth."""
        base_state.push("step 1", temperature=0.6)
        base_state.push("step 2", temperature=0.5)
        
        branch = base_state.branch_from(depth=1)
        
        assert branch is not None
        assert branch.depth == 2  # prompt + step 1
        assert branch.current == "step 1"
    
    def test_branch_does_not_affect_original(self, base_state):
        """Branching does not modify original state."""
        base_state.push("step 1")
        base_state.push("step 2")
        
        original_depth = base_state.depth
        
        branch = base_state.branch_from(depth=1)
        branch.push("step_2_alternative")
        
        assert base_state.depth == original_depth
        assert "step_2_alternative" not in base_state.history
    
    def test_branch_from_exceeding_depth_returns_none(self, base_state):
        """Branch from invalid depth returns None."""
        branch = base_state.branch_from(depth=10)  # Only 1 step exists
        
        assert branch is None


# ─────────────────────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_cannot_revert_below_prompt(self, base_state):
        """Cannot revert below initial prompt."""
        base_state.revert()  # Try to revert prompt
        base_state.revert()  # Try again
        
        assert base_state.depth == 1
        assert base_state.current == "Solve this math problem"
    
    def test_empty_history_after_init(self, base_state):
        """History is never empty - always contains prompt."""
        assert len(base_state.history) >= 1
    
    def test_current_always_matches_history_last(self, base_state):
        """Current always matches last history entry."""
        base_state.push("step 1")
        base_state.push("step 2")
        
        assert base_state.current == base_state.history[-1]
        
        base_state.revert()
        
        assert base_state.current == base_state.history[-1]
    
    def test_metrics_default_to_empty_dict(self, base_state):
        """Metrics default to empty dict if not provided."""
        base_state.push("step 1")
        
        assert isinstance(base_state.metrics, dict)
        assert len(base_state.metrics) == 0


# ─────────────────────────────────────────────────────────────
# INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests simulating real usage patterns."""
    
    def test_hallucination_recovery_pattern(self, base_state):
        """
        Simulates the critical hallucination recovery pattern.
        This is the main use case for the stack-based fix.
        """
        # Normal reasoning
        base_state.push("step 1: analyze problem", temperature=0.7)
        base_state.push("step 2: initial approach", temperature=0.7)
        
        # Hallucination detected
        base_state.push("step 3: hallucinated_content", temperature=0.7)
        
        assert base_state.current == "step 3: hallucinated_content"
        
        # Recovery: revert removes hallucinated state
        base_state.revert()
        
        assert base_state.current == "step 2: initial approach"
        assert "hallucinated_content" not in base_state.history
        
        # Retry with lower temperature
        base_state.temperature = 0.5
        base_state.push("step 3: corrected_approach", temperature=0.5)
        
        assert base_state.current == "step 3: corrected_approach"
        assert base_state.temperature == 0.5
    
    def test_multiple_recovery_cycles(self, base_state):
        """Tests multiple hallucination recovery cycles."""
        for i in range(3):
            base_state.push(f"step_{i}_good")
            base_state.push(f"step_{i}_bad")
            base_state.revert()  # Remove bad state
            
            assert f"step_{i}_bad" not in base_state.history
        
        assert base_state.depth == 4  # prompt + 3 good steps
    
    def test_reset_after_multiple_reverts(self, base_state):
        """Tests reset after multiple revert operations."""
        base_state.push("step 1")
        base_state.push("step 2")
        base_state.push("step 3")
        
        base_state.revert()
        base_state.revert()
        
        assert base_state.depth == 2
        
        base_state.reset()
        
        assert base_state.depth == 1
        assert base_state.current == "Solve this math problem"


# ─────────────────────────────────────────────────────────────
# DEBUGGING UTILITIES
# ─────────────────────────────────────────────────────────────

class TestDebugUtilities:
    """Test debugging and inspection utilities."""
    
    def test_debug_dump_produces_output(self, base_state):
        """Debug dump produces readable output."""
        base_state.push("step 1", temperature=0.6)
        
        dump = base_state.debug_dump()
        
        assert isinstance(dump, str)
        assert "step 1" in dump
        assert "0.60" in dump
    
    def test_to_dict_serializes_state(self, base_state):
        """to_dict produces serializable representation."""
        base_state.push("step 1", temperature=0.6)
        
        state_dict = base_state.to_dict()
        
        assert isinstance(state_dict, dict)
        assert state_dict["current"] == "step 1"
        assert state_dict["temperature"] == 0.6
        assert state_dict["depth"] == 2


# ─────────────────────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])