"""
Energy control policy - deterministic controller mapping.

Policy responds to dynamics but does not compute them.
"""
from dataclasses import dataclass
from .types import EnergyDynamics, ControlDecision, Action, ViolationLevel


@dataclass
class EnergyControlPolicy:
    """
    Graded energy controller.
    
    Maps violation + slope → action + temperature.
    Deterministic: same inputs → same outputs.
    """
    min_temperature: float = 0.1
    cooldown_medium: float = 0.85
    cooldown_hard_small_slope: float = 0.7
    cooldown_hard_large_slope: float = 0.5
    runaway_slope_eps: float = 0.05
    
    def evaluate(
        self,
        dyn: EnergyDynamics,
        temperature: float,
    ) -> ControlDecision:
        """
        Evaluate dynamics and return control decision.
        
        Policy does NOT:
        - Compute slope
        - Compute violation
        - Define collapse
        - Mutate counters
        
        Policy only responds to pre-computed dynamics.
        """
        violation = dyn.violation
        slope = dyn.slope
        
        # No violation → allow
        if violation == ViolationLevel.NONE:
            return ControlDecision(
                action=Action.ALLOW,
                new_temperature=temperature,
                reason=f"No violation (energy={dyn.energy:.3f})",
            )
        
        # Soft violation → allow (monitoring only)
        if violation == ViolationLevel.SOFT:
            return ControlDecision(
                action=Action.ALLOW,
                new_temperature=temperature,
                reason=f"Soft violation (energy={dyn.energy:.3f})",
            )
        
        # Medium violation → rollback + cool
        if violation == ViolationLevel.MEDIUM:
            if slope <= 0:
                # Self-correcting → allow
                return ControlDecision(
                    action=Action.ALLOW,
                    new_temperature=temperature,
                    reason=f"Medium violation but self-correcting (slope={slope:.3f})",
                )
            
            new_temp = max(self.min_temperature, temperature * self.cooldown_medium)
            return ControlDecision(
                action=Action.ROLLBACK,
                new_temperature=new_temp,
                reason=f"Medium violation + rising (slope={slope:.3f})",
            )
        
        # Hard violation → escalate based on slope
        if violation == ViolationLevel.HARD:
            if slope <= 0:
                # High but not rising → rollback
                new_temp = max(self.min_temperature, temperature * self.cooldown_hard_small_slope)
                return ControlDecision(
                    action=Action.ROLLBACK,
                    new_temperature=new_temp,
                    reason=f"Hard violation but not rising (slope={slope:.3f})",
                )
            
            if slope < self.runaway_slope_eps:
                # Small slope → rollback
                new_temp = max(self.min_temperature, temperature * self.cooldown_hard_small_slope)
                return ControlDecision(
                    action=Action.ROLLBACK,
                    new_temperature=new_temp,
                    reason=f"Hard violation, small slope (slope={slope:.3f})",
                )
            else:
                # Large slope → reset (runaway)
                new_temp = max(self.min_temperature, temperature * self.cooldown_hard_large_slope)
                return ControlDecision(
                    action=Action.RESET,
                    new_temperature=new_temp,
                    reason=f"Hard violation, runaway slope (slope={slope:.3f})",
                )
        
        # Fallback
        return ControlDecision(
            action=Action.ALLOW,
            new_temperature=temperature,
            reason="Fallback allow",
        )