class RegimeCollapseDetector:

    def detect(
        self,
        collapse_probs: list[float],
        energies: list[float],
        tau_stable: float,
        tau_unstable: float,
        tau_soft: float,
        tau_medium: float,
    ):
        """
        Returns first collapse time or None if censored.
        """

        entered_unstable_at = None

        # Step 1: detect entry
        for t in range(len(collapse_probs)):
            if (
                collapse_probs[t] > tau_unstable
                and energies[t] > tau_medium
            ):
                entered_unstable_at = t
                break

        if entered_unstable_at is None:
            return None

        # Step 2: check for recovery
        for t in range(entered_unstable_at + 1, len(collapse_probs)):
            if (
                collapse_probs[t] < tau_stable
                and energies[t] < tau_soft
            ):
                return None  # temporary instability

        return entered_unstable_at