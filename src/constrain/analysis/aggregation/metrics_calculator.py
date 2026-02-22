# metrics_calculator.py

import re
import unicodedata
from typing import Dict, Optional


class MetricsCalculator:

    # -------------------------------------------------
    # TEXT SIGNALS
    # -------------------------------------------------

    PHASE_MAP = {
        "stable": 0,
        "drift": 1,
        "unstable": 2,
        "collapse": 3,
    }


    @staticmethod
    def foreign_char_ratio(text: str) -> float:
        """YOUR CRITICAL OBSERVATION: Non-ASCII *LETTERS* (Chinese, Arabic) precede collapse.
        Counts ONLY letters outside ASCII range - ignores emojis/punctuation/symbols."""
        if not text:
            return 0.0
        foreign_letters = sum(
            1 for c in text 
            if ord(c) > 127 and unicodedata.category(c).startswith('L')  # 'L' = Letter
        )
        return foreign_letters / len(text) if text else 0.0

    @staticmethod
    def ascii_ratio(text: str) -> float:
        if not text:
            return 1.0
        ascii_chars = sum(1 for c in text if ord(c) <= 127)
        return ascii_chars / len(text)

    @staticmethod
    def repetition_score(text: str) -> float:
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 3:
            return 0.0
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        if not trigrams:
            return 0.0
        return 1.0 - (len(set(trigrams)) / len(trigrams))

    # -------------------------------------------------
    # PHASE CLASSIFIER
    # -------------------------------------------------

    @staticmethod
    def compute_phase(energy, tau_soft, tau_medium, tau_hard):
        if energy < tau_soft:
            return 1 # "stable"
        elif energy < tau_medium:
            return 2 # "drift"
        elif energy < tau_hard:
            return 3 # "unstable"
        else:
            return 4 # "collapse"

    PHASE_VALUE_TO_LABEL = {v: k for k, v in PHASE_MAP.items()}

    @staticmethod
    def compute_phase_label(energy, tau_soft, tau_medium, tau_hard):
        if energy < tau_soft:
            return "stable"
        elif energy < tau_medium:
            return "drift"
        elif energy < tau_hard:
            return "unstable"
        else:
            return "collapse"

    # -------------------------------------------------
    # ACCURACY
    # -------------------------------------------------

    @staticmethod
    def compute_accuracy(reasoning, gold_answer):
        numbers = re.findall(r"-?\d+\.?\d*", reasoning)
        extracted = numbers[-1] if numbers else None
        correctness = extracted == gold_answer.strip() if extracted else False

        return {
            "accuracy": 1.0 if correctness else 0.0,
            "correctness": 1 if correctness else 0,
            "extracted_answer": extracted,
        }

    # -------------------------------------------------
    # FULL METRIC PACK
    # -------------------------------------------------

    @classmethod
    def compute_all(
        cls,
        reasoning: str,
        gold_answer: Optional[str],
        energy_metrics: Dict[str, float],
        cfg,
    ) -> Dict[str, float]:

        text_metrics = {
            "foreign_char_ratio": cls.foreign_char_ratio(reasoning),
            "ascii_ratio": cls.ascii_ratio(reasoning),
            "repetition_score": cls.repetition_score(reasoning),
        }

        # For open-ended long-horizon conversations, we often have no gold answers.
        # In that case, skip accuracy and leave placeholders.
        if gold_answer is None:
            accuracy_metrics = {
                "accuracy": float("nan"),
                "correctness": None,
                "extracted_answer": None,
            }
        else:
            accuracy_metrics = cls.compute_accuracy(reasoning, gold_answer)

        energy = energy_metrics.get("value", 0.0)
        phase_label = cls.compute_phase_label(
            energy,
            cfg.tau_soft,
            cfg.tau_medium,
            cfg.tau_hard,
        )

        phase_value = cls.PHASE_MAP[phase_label]

        return {
            **energy_metrics,
            **text_metrics,
            **accuracy_metrics,
            "phase_value": phase_value,
        }
