"""Environment adapter for long-horizon conversation experiments.

Key property: **preview/commit** separation.

We must never accidentally advance the conversation turn counter when a REVERT
regenerates a response. Survival time is defined in *turns until failure*, so
we generate candidate outputs with :meth:`preview_turn` and only advance state
with :meth:`commit_turn`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from constrain.simulator import SingleStepContext, run_single_step


@dataclass
class ConversationState:
    prompt_text: str
    history: List[Dict[str, str]] = field(default_factory=list)
    turn: int = 0


class ConversationEnv:
    def __init__(
        self,
        *,
        seed: int = 0,
        max_tokens: int = 512,
        base_temperature: float = 0.7,
        safe_temperature: float = 0.0,
        learned_model_path: str | None = None,
    ):
        self.seed = seed
        self.max_tokens = max_tokens
        self.base_temperature = base_temperature
        self.safe_temperature = safe_temperature
        self.temperature = base_temperature
        self.ctx = SingleStepContext.build(learned_model_path=learned_model_path)

    def reset(self, prompt_text: str) -> ConversationState:
        return ConversationState(
            prompt_text=prompt_text,
            history=[{"role": "user", "content": prompt_text}],
            turn=0,
        )

    def preview_turn(self, state: ConversationState) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
        """Generate a candidate response + features without committing."""
        step = run_single_step(
            prompt_text=state.prompt_text,
            history=state.history,
            policy_action="ACCEPT",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            gold_answer=None,
            ctx=self.ctx,
        )
        raw = step.model_dump()

        # Trust the StepDTO as the *single source of truth*.
        if raw.get("collapse_probability") is None:
            raise ValueError("StepDTO missing collapse_probability; learned head not wired")

        feats = {
            "collapse_prob": float(raw["collapse_probability"]),
            "total_energy": float(raw.get("total_energy", 0.0)),
            "accuracy": float(raw.get("accuracy")) if raw.get("accuracy") is not None else float("nan"),
            "entropy": float(raw.get("entropy") or 0.0),
            "token_count": float(raw.get("token_count") or 0.0),
            "repetition_score": float(raw.get("repetition_score") or 0.0),
            "delta_value": float(raw.get("delta_value") or 0.0),
        }
        return step.reasoning_text, raw, feats

    def commit_turn(self, state: ConversationState, response: str) -> ConversationState:
        """Commit the chosen assistant response to history and advance one turn."""
        return ConversationState(
            prompt_text=state.prompt_text,
            history=state.history + [{"role": "assistant", "content": response}],
            turn=state.turn + 1,
        )

    def truncate_context(self, state: ConversationState, *, truncate_turns: int, min_history_msgs: int = 1) -> ConversationState:
        """Drop the last N (user+assistant) turns worth of messages.

        We define a "turn" as a user message followed by an assistant message.
        In our histories during survival tests, we typically append only the assistant.
        To keep semantics simple, this function just drops the last ``2*truncate_turns``
        messages if possible, while keeping at least ``min_history_msgs`` messages.
        """
        if truncate_turns <= 0:
            return state

        drop = 2 * truncate_turns
        if len(state.history) <= min_history_msgs:
            return state
        new_len = max(min_history_msgs, len(state.history) - drop)
        return ConversationState(
            prompt_text=state.prompt_text,
            history=state.history[:new_len],
            turn=max(0, state.turn - truncate_turns),
        )
