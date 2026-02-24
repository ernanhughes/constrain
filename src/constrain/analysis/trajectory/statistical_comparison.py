from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """
    Result of bootstrap statistical comparison.
    
    Attributes:
        mean_diff: Mean difference (treatment - baseline)
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
        significant: True if CI does not cross zero
        n_baseline: Sample size for baseline group
        n_treatment: Sample size for treatment group
        p_value: Approximate p-value (optional, from permutation)
    """
    mean_diff: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_baseline: int
    n_treatment: int
    p_value: Optional[float] = None


class BootstrapComparator:
    """
    Computes statistical significance using bootstrap resampling.
    
    Supports:
    - Difference in means (accuracy, energy, etc.)
    - Difference in proportions (intervention rate, collapse rate)
    - AUC comparison (for signal discovery)
    - Permutation tests for p-values (optional)
    
    All methods are non-parametric and make no distributional assumptions.
    """

    def __init__(self, seed: int = 42, n_iterations: int = 2000):
        """
        Initialize comparator.
        
        Args:
            seed: Random seed for reproducibility
            n_iterations: Number of bootstrap samples (more = more precise, slower)
        """
        self.seed = seed
        self.n_iterations = n_iterations
        self.rng = np.random.RandomState(seed)

    def compare_means(
        self,
        baseline: List[float],
        treatment: List[float],
        confidence_level: float = 0.95,
        compute_p_value: bool = False,
    ) -> BootstrapResult:
        """
        Compare means of two groups using bootstrap.
        
        Args:
            baseline: List of metric values for baseline group
            treatment: List of metric values for treatment group
            confidence_level: Confidence level for interval (default 0.95)
            compute_p_value: If True, also compute approximate p-value via permutation
            
        Returns:
            BootstrapResult with mean difference and confidence interval
        """
        if len(baseline) == 0 or len(treatment) == 0:
            return BootstrapResult(
                mean_diff=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                significant=False,
                n_baseline=len(baseline),
                n_treatment=len(treatment),
            )

        baseline = np.array(baseline, dtype=float)
        treatment = np.array(treatment, dtype=float)

        # Observed difference
        observed_diff = treatment.mean() - baseline.mean()

        # Bootstrap resampling
        diffs = []
        for _ in range(self.n_iterations):
            idx_b = self.rng.choice(len(baseline), len(baseline), replace=True)
            idx_t = self.rng.choice(len(treatment), len(treatment), replace=True)
            diff = treatment[idx_t].mean() - baseline[idx_b].mean()
            diffs.append(diff)

        diffs = np.array(diffs)

        # Confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(diffs, 100 * alpha / 2)
        ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))

        # Significance: CI doesn't cross zero
        significant = (ci_lower > 0) or (ci_upper < 0)

        # Optional p-value via permutation test
        p_value = None
        if compute_p_value:
            p_value = self._permutation_p_value(baseline, treatment, observed_diff)

        return BootstrapResult(
            mean_diff=float(observed_diff),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            significant=significant,
            n_baseline=len(baseline),
            n_treatment=len(treatment),
            p_value=p_value,
        )

    def compare_proportions(
        self,
        baseline_successes: int,
        baseline_total: int,
        treatment_successes: int,
        treatment_total: int,
        confidence_level: float = 0.95,
    ) -> BootstrapResult:
        """
        Compare proportions (e.g., accuracy, intervention rate) using bootstrap.
        
        Args:
            baseline_successes: Number of successes in baseline
            baseline_total: Total samples in baseline
            treatment_successes: Number of successes in treatment
            treatment_total: Total samples in treatment
            
        Returns:
            BootstrapResult with proportion difference and confidence interval
        """
        # Convert to binary arrays for bootstrap
        baseline = np.array([1] * baseline_successes + [0] * (baseline_total - baseline_successes))
        treatment = np.array([1] * treatment_successes + [0] * (treatment_total - treatment_successes))

        return self.compare_means(
            baseline.tolist(),
            treatment.tolist(),
            confidence_level=confidence_level,
        )

    def compare_aucs(
        self,
        baseline_aucs: List[float],
        treatment_aucs: List[float],
        confidence_level: float = 0.95,
    ) -> BootstrapResult:
        """
        Compare AUC scores across cross-validation folds.
        
        Args:
            baseline_aucs: List of AUC values for baseline (one per fold)
            treatment_aucs: List of AUC values for treatment (one per fold)
            
        Returns:
            BootstrapResult with AUC difference and confidence interval
        """
        return self.compare_means(
            baseline_aucs,
            treatment_aucs,
            confidence_level=confidence_level,
        )

    def _permutation_p_value(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
        observed_diff: float,
        n_permutations: int = 1000,
    ) -> float:
        """
        Compute approximate p-value via permutation test.
        
        Null hypothesis: both groups come from same distribution.
        """
        combined = np.concatenate([baseline, treatment])
        n_b = len(baseline)

        extreme_count = 0
        for _ in range(n_permutations):
            self.rng.shuffle(combined)
            perm_b = combined[:n_b]
            perm_t = combined[n_b:]
            perm_diff = perm_t.mean() - perm_b.mean()

            if abs(perm_diff) >= abs(observed_diff):
                extreme_count += 1

        return float(extreme_count / n_permutations)

    def format_result(self, result: BootstrapResult, metric_name: str = "metric") -> str:
        """
        Format BootstrapResult as human-readable string.
        
        Example:
            "Accuracy: +3.2% [95% CI: -0.5%, +6.9%] (not significant)"
        """
        if np.isnan(result.mean_diff):
            return f"{metric_name}: insufficient data"

        sign = "+" if result.mean_diff >= 0 else ""
        pct = result.mean_diff * 100
        ci_low = result.ci_lower * 100
        ci_high = result.ci_upper * 100

        sig_text = "significant" if result.significant else "not significant"

        return f"{metric_name}: {sign}{pct:.2f}% [95% CI: {ci_low:.2f}%, {ci_high:.2f}%] ({sig_text})"