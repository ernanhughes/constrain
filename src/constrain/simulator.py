
"""Single-step generation utilities.

This module exists to support *evaluation* experiments (e.g. survival analysis)
that need to execute exactly one conversational turn at a time.

It intentionally does **not** write to the DB; the caller can persist the
returned StepDTO if desired.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any, Dict, List, Optional

from constrain.analysis.aggregation.metrics_calculator import MetricsCalculator
from constrain.config import get_config
from constrain.data.schemas.step import StepDTO
from constrain.model import call_model
from constrain.energy_computer import compute_energy
from constrain.policy.learned_policy import LearnedPolicy


def history_to_prompt(history: List[Dict[str, str]]) -> str:
    """Convert chat-style messages to a single prompt string.

    Your core stack is prompt-string based. Survival experiments operate on
    chat histories, so we need a deterministic serializer.
    """

    parts: List[str] = []
    for msg in history:
        role = (msg.get("role") or "user").strip().lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


@dataclass
class SingleStepContext:
    """Caches heavyweight objects for repeated single-step calls."""

    learned_policy: Optional[LearnedPolicy]

    @staticmethod
    def build(*, learned_model_path: Optional[str] = None) -> "SingleStepContext":
        cfg = get_config()
        model_path = learned_model_path or getattr(cfg, "learned_model_path", None)
        learned = LearnedPolicy(model_path=model_path, threshold=getattr(cfg, "learned_policy_threshold", 0.5)) if model_path else None
        return SingleStepContext(learned_policy=learned)


def run_single_step(
    *,
    prompt_text: str,
    history: List[Dict[str, str]],
    policy_action: str = "ACCEPT",
    temperature: float = 0.7,
    max_tokens: int = 512,
    gold_answer: Optional[str] = None,
    ctx: Optional[SingleStepContext] = None,
) -> StepDTO:
    """Execute one conversational turn and return a populated StepDTO.

    - Uses :func:`call_model` (prompt-string based) with a serialized history.
    - Computes energy via :func:`constrain.energy_computer.compute_energy`.
    - Computes text metrics + (optional) accuracy via :class:`MetricsCalculator`.
    - Computes ``collapse_probability`` using the learned policy model (if available).
    """

    cfg = get_config()
    ctx = ctx or SingleStepContext.build()

    # -----------------------------
    # 1) Generate model output
    # -----------------------------
    prompt = history_to_prompt(history)
    response = call_model(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # -----------------------------
    # 2) Compute energy / geometry signals
    # -----------------------------
    reasoning_history = [m["content"] for m in history if (m.get("role") or "").lower() == "assistant" and (m.get("content") or "").strip()]
    energy_metrics = compute_energy(prompt_text, response, reasoning_history)

    # -----------------------------
    # 3) Compute metrics pack
    # -----------------------------
    metrics = MetricsCalculator.compute_all(
        reasoning=response,
        gold_answer=gold_answer,
        energy_metrics=energy_metrics,
        cfg=cfg,
    )

    # -----------------------------
    # 4) Collapse probability (learned head)
    # -----------------------------
    collapse_prob: Optional[float] = None
    feat: Dict[str, float] = {}
    if ctx.learned_policy is not None:
        # Use the same feature schema as in training.
        feat = {
            "iteration": float(len([m for m in history if (m.get("role") or "").lower() == "assistant"])) ,
            "temperature": float(temperature),
            "ascii_ratio": float(metrics.get("ascii_ratio", 1.0)),
            "foreign_char_ratio": float(metrics.get("foreign_char_ratio", 0.0)),
            "repetition_score": float(metrics.get("repetition_score", 0.0)),
            "total_energy": float(metrics.get("total_energy", 0.0)),
            "grounding_energy": float(metrics.get("grounding_energy", 0.0)),
            "stability_energy": float(metrics.get("stability_energy", 0.0)),
        }
        _, collapse_prob = ctx.learned_policy.decide(feat)

    # -----------------------------
    # 5) Return StepDTO
    # -----------------------------
    return StepDTO(
        id=None,
        run_id=-1,
        problem_id=str(prompt_text)[:64],
        iteration=int(feat.get("iteration", 0.0)),
        prompt_text=prompt_text,
        reasoning_text=response,
        accuracy=float(metrics.get("accuracy", float("nan"))),
        total_energy=float(metrics.get("total_energy", 0.0)),
        collapse_probability=float(collapse_prob) if collapse_prob is not None else float("nan"),
        policy_action=policy_action,
        phase=MetricsCalculator.compute_phase_label(
            float(metrics.get("total_energy", 0.0)),
            cfg.tau_soft,
            cfg.tau_medium,
            cfg.tau_hard,
        ),
        timestamp=float(time()),
        entropy=float(metrics.get("entropy", 0.0)) if metrics.get("entropy") is not None else None,
        token_count=float(metrics.get("token_count", 0.0)) if metrics.get("token_count") is not None else None,
        repetition_score=float(metrics.get("repetition_score", 0.0)) if metrics.get("repetition_score") is not None else None,
    )
