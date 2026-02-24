# constrain/reasoning_state.py

from constrain.config import get_config
from typing import List, Optional, Dict
from dataclasses import dataclass
import time


@dataclass
class StateSnapshot:
    """Immutable snapshot of a single reasoning step."""
    content: str
    temperature: float
    metrics: Dict[str, float]


class ReasoningState:
    """
    Stack-based reasoning state machine WITH PERSISTENCE.
    
    Key properties:
    - True push/pop semantics (revert actually removes bad states)
    - Temperature tracked per step
    - Metrics snapshot preserved for analysis
    - Auto-snapshot to database for replay
    - Control system fields persisted for analysis
    """
    
    def __init__(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        run_id: Optional[str] = None,
        problem_id: Optional[int] = None,
        snapshot_store=None,
    ):
        cfg = get_config()
        
        self.prompt = prompt
        self._initial_temperature = (
            temperature 
            if temperature is not None 
            else cfg.initial_temperature
        )
        
        # Stack of StateSnapshot objects
        self._stack: List[StateSnapshot] = [
            StateSnapshot(
                content=prompt,
                temperature=self._initial_temperature,
                metrics={}
            )
        ]
        
        # Persistence
        self._run_id = run_id
        self._problem_id = problem_id
        self._snapshot_store = snapshot_store
        self._iteration = 0
        self._attempt = 0
        self._last_snapshot_id = None

    # ─────────────────────────────────────────────────────────────
    # READ-ONLY PROPERTIES
    # ─────────────────────────────────────────────────────────────
    
    @property
    def current(self) -> str:
        return self._stack[-1].content
    
    @property
    def temperature(self) -> float:
        return self._stack[-1].temperature
    
    @temperature.setter
    def temperature(self, value: float):
        """
        Update temperature on current stack top.
        
        This allows runner.py to do: state.temperature = new_temp
        """
        if len(self._stack) > 0:
            # Create new snapshot with updated temperature
            old_snapshot = self._stack[-1]
            self._stack[-1] = StateSnapshot(
                content=old_snapshot.content,
                temperature=value,
                metrics=old_snapshot.metrics
            )
    
    @property
    def history(self) -> List[str]:
        return [s.content for s in self._stack]
    
    @property
    def depth(self) -> int:
        return len(self._stack)
    
    @property
    def metrics(self) -> Dict[str, float]:
        return self._stack[-1].metrics

    # ─────────────────────────────────────────────────────────────
    # STATE MUTATION (PUSH/POP SEMANTICS)
    # ─────────────────────────────────────────────────────────────
    
    def push(
        self, 
        content: str, 
        temperature: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        policy_action: str = "ACCEPT",
        total_energy: Optional[float] = None,
        grounding_energy: Optional[float] = None,
        stability_energy: Optional[float] = None,
        # Control system fields
        energy_slope: Optional[float] = None,
        violation_level: Optional[str] = None,
        consecutive_hard: Optional[int] = None,
        consecutive_rising: Optional[int] = None,
        collapse_flag: bool = False,
        attempt: Optional[int] = None,
        iteration: Optional[int] = None,
    ) -> None:
        """
        Push new state onto stack AND persist snapshot.
        
        Control system fields are passed through to persistence layer.
        """
        new_temp = temperature if temperature is not None else self.temperature
        new_metrics = metrics if metrics is not None else {}
        
        self._stack.append(StateSnapshot(
            content=content,
            temperature=new_temp,
            metrics=new_metrics
        ))
        
        # Update counters (use provided values or increment)
        self._attempt = attempt if attempt is not None else self._attempt + 1
        self._iteration = iteration if iteration is not None else self._iteration + 1
        
        # Persist snapshot with all control fields
        self._persist_snapshot(
            policy_action=policy_action,
            total_energy=total_energy,
            grounding_energy=grounding_energy,
            stability_energy=stability_energy,
            energy_slope=energy_slope,
            violation_level=violation_level,
            consecutive_hard=consecutive_hard,
            consecutive_rising=consecutive_rising,
            collapse_flag=collapse_flag,
        )
    
    def accept(self, reasoning: str, **kwargs) -> None:
        """Alias for push() for backward compatibility."""
        self.push(reasoning, policy_action="ACCEPT", **kwargs)
    
    def pop(self, steps: int = 1) -> bool:
        """
        Actually remove bad states from stack.
        
        CRITICAL FIX for runaway hallucination recovery.
        """
        if len(self._stack) <= 1:
            return False
        
        steps_to_pop = min(steps, len(self._stack) - 1)
        
        for _ in range(steps_to_pop):
            self._stack.pop()
        
        # Persist snapshot marking revert
        self._persist_snapshot(
            policy_action="REVERT",
            is_after_revert=True,
        )
        
        return True
    
    def revert(self, steps: int = 1) -> bool:
        """Alias for pop() for backward compatibility."""
        return self.pop(steps)
    
    def reset(self) -> None:
        """Reset to initial prompt, clearing all reasoning history."""
        initial_temp = self._initial_temperature
        self._stack = [
            StateSnapshot(
                content=self.prompt,
                temperature=initial_temp,
                metrics={}
            )
        ]
        
        # Reset counters
        self._iteration = 0
        self._attempt += 1
        
        # Persist snapshot marking reset
        self._persist_snapshot(
            policy_action="RESET",
            is_after_reset=True,
        )

    def branch_from(self, depth: int = 1) -> Optional["ReasoningState"]:
        """
        Create a new state branched from N steps back.
        
        Args:
            depth: How many steps back to branch from (1 = current, 2 = previous, etc.)
        
        Returns:
            New ReasoningState instance, or None if depth exceeds stack
        """
        if depth >= len(self._stack):
            return None
        
        # Create independent clone
        cloned = ReasoningState.__new__(ReasoningState)
        cloned.prompt = self.prompt
        cloned._initial_temperature = self._initial_temperature
        cloned._stack = [
            StateSnapshot(s.content, s.temperature, dict(s.metrics)) 
            for s in self._stack[:-depth]
        ]
        cloned._run_id = None  # Branch doesn't inherit run tracking
        cloned._problem_id = None
        cloned._snapshot_store = None
        cloned._iteration = 0
        cloned._attempt = 0
        cloned._last_snapshot_id = None
        
        return cloned

    # ─────────────────────────────────────────────────────────────
    # PERSISTENCE HELPERS
    # ─────────────────────────────────────────────────────────────
    
    def _persist_snapshot(
        self,
        policy_action: str = "ACCEPT",
        is_after_revert: bool = False,
        is_after_reset: bool = False,
        total_energy: Optional[float] = None,
        grounding_energy: Optional[float] = None,
        stability_energy: Optional[float] = None,
        # Control system fields
        energy_slope: Optional[float] = None,
        violation_level: Optional[str] = None,
        consecutive_hard: Optional[int] = None,
        consecutive_rising: Optional[int] = None,
        collapse_flag: bool = False,
    ) -> None:
        """
        Persist current state to database for replay.
        
        All control system fields are persisted for analysis.
        """
        if self._snapshot_store is None or self._run_id is None:
            return  # Skip if no persistence configured
        
        from constrain.data.schemas.reasoning_state import ReasoningStateSnapshotDTO
        
        dto = ReasoningStateSnapshotDTO(
            run_id=self._run_id,
            problem_id=self._problem_id or 0,
            step_id=None,
            iteration=self._iteration,
            prompt_text=self.prompt,
            current_reasoning=self.current,
            history=self.history,
            temperature=self.temperature,
            initial_temperature=self._initial_temperature,
            stack_depth=self.depth,
            is_after_revert=is_after_revert,
            is_after_reset=is_after_reset,
            parent_snapshot_id=self._last_snapshot_id,
            total_energy=total_energy,
            grounding_energy=grounding_energy,
            stability_energy=stability_energy,
            policy_action=policy_action,
            policy_action_reason=None,
            created_at=time.time(),
            # Control system fields
            attempt=self._attempt,
            energy_slope=energy_slope,
            violation_level=violation_level,
            consecutive_hard=consecutive_hard,
            consecutive_rising=consecutive_rising,
            collapse_flag=collapse_flag,
        )
        
        try:
            persisted = self._snapshot_store.create(dto)
            self._last_snapshot_id = persisted.id
        except Exception as e:
            import logging
            logging.warning(f"Failed to persist state snapshot: {e}")

    # ─────────────────────────────────────────────────────────────
    # REPLAY HELPERS
    # ─────────────────────────────────────────────────────────────
    
    def to_dict(self) -> dict:
        """Serialize state for inspection/debugging."""
        return {
            "prompt": self.prompt,
            "current": self.current,
            "history": self.history,
            "temperature": self.temperature,
            "depth": self.depth,
            "iteration": self._iteration,
            "attempt": self._attempt,
        }
    
    def debug_dump(self) -> str:
        """Return detailed stack state for debugging."""
        lines = [f"ReasoningState (depth={self.depth}, iteration={self._iteration}, attempt={self._attempt})"]
        lines.append(f"  Prompt: {self.prompt[:50]}...")
        lines.append(f"  Current temp: {self.temperature}")
        lines.append("  Stack:")
        for i, snapshot in enumerate(self._stack):
            content_preview = snapshot.content[:40].replace("\n", " ")
            lines.append(
                f"    [{i}] temp={snapshot.temperature:.2f} | {content_preview}..."
            )
        return "\n".join(lines)