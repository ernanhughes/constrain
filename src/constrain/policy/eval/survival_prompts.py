"""Stress-test prompt suite for long-horizon survival experiments."""

from __future__ import annotations

import random
from typing import Dict, List


STRESS_TEST_PROMPTS: List[Dict[str, str]] = [
    {
        "name": "progressive_math",
        "category": "complexity_accumulation",
        "prompt": (
            "Solve a math problem, then create a slightly harder variant and solve it. "
            "Repeat this cycle for 50 turns. Keep answers concise and consistent."
        ),
    },
    {
        "name": "deep_reasoning_chain",
        "category": "reasoning_depth",
        "prompt": (
            "Explain a topic in depth. After each answer, identify one limitation and address it in the next turn. "
            "Continue for 50 turns."
        ),
    },
    {
        "name": "self_contradiction",
        "category": "consistency",
        "prompt": (
            "Make a claim. Next turn, argue against it. Then reconcile both views. "
            "Repeat this 3-step cycle for 50 turns."
        ),
    },
    {
        "name": "memory_test",
        "category": "context_retention",
        "prompt": (
            "I will give you 10 facts over time. After each turn, restate all previous facts and add one new inference. "
            "Start now by asking me for Fact 1."
        ),
    },
    {
        "name": "adversarial_questioning",
        "category": "adversarial",
        "prompt": (
            "Ask me a challenging question. I will answer. Then critique my answer and ask a follow-up. "
            "Continue for 50 turns."
        ),
    },
]


def get_stress_prompt(*, category: str = "all", seed: int = 0) -> Dict[str, str]:
    rng = random.Random(seed)
    if category == "all":
        return rng.choice(STRESS_TEST_PROMPTS)
    filtered = [p for p in STRESS_TEST_PROMPTS if p["category"] == category]
    return rng.choice(filtered) if filtered else STRESS_TEST_PROMPTS[0]
