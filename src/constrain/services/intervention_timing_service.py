"""constrain.services.intervention_timing_service
Intervention Timing Analysis Service

Determines whether interventions occur:
- Too late (collapse at t+1)
- Too early
- At optimal margin
- Or randomly

Returns distributional statistics, not just prints.

Production features:
- Config-driven thresholds and parameters
- Structured logging and diagnostics
- Export to JSON/CSV for downstream analysis
- Graceful handling of edge cases
- Type safety and validation
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.utils.json_utils import dumps_safe

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Configuration Dataclass
# ─────────────────────────────────────────────────────────────
@dataclass
class TimingAnalysisConfig:
    """Configuration for intervention timing analysis."""
    
    # Thresholds
    tau_hard_override: Optional[float] = None  # Override config tau_hard if specified
    
    # Timing buckets for verdict
    too_late_threshold: int = 1  # Interventions ≤1 step before collapse = "too late"
    early_enough_threshold: int = 3  # Interventions ≥3 steps before = "early enough"
    early_enough_pct: float = 40.0  # Min % for "early enough" verdict
    too_late_pct: float = 50.0  # Min % for "too late" verdict
    
    # Histogram bins
    histogram_max_delta: int = 10
    
    # Export options
    export_results: bool = True
    export_path: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, config: Dict) -> "TimingAnalysisConfig":
        """Create config from dictionary (YAML/JSON)."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict:
        """Serialize config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────
# Result Dataclass
# ─────────────────────────────────────────────────────────────
@dataclass
class TimingAnalysisResult:
    """Structured result for intervention timing analysis."""
    
    run_id: str
    status: str  # "success", "skipped", "error"
    
    # Core metrics
    mean_delta: Optional[float]
    median_delta: Optional[float]
    std_delta: Optional[float]
    
    # Distribution metrics
    pct_delta_le_1: Optional[float]  # % interventions ≤1 step before collapse
    pct_delta_le_2: Optional[float]  # % interventions ≤2 steps before collapse
    pct_delta_ge_3: Optional[float]  # % interventions ≥3 steps before collapse
    
    # Counts
    n_interventions: int
    n_collapsed_problems: int
    n_analyzed_interventions: int
    
    # Histogram: delta -> count
    histogram: Dict[int, int]
    
    # Verdict
    verdict: str
    
    # Diagnostics (for debugging)
    diagnostics: Dict[str, Any]
    
    # Metadata
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# ─────────────────────────────────────────────────────────────
# Main Service Class
# ─────────────────────────────────────────────────────────────
class InterventionTimingService:
    """
    Analyzes timing of policy interventions relative to collapse events.
    
    Core methodology:
    - Detect first collapse iteration per problem (energy > tau_hard)
    - Find all interventions (policy_action != "ACCEPT")
    - Compute delta = collapse_iteration - intervention_iteration
    - Analyze distribution of deltas to assess intervention timing quality
    
    Usage:
        service = InterventionTimingService(memory, config=TimingAnalysisConfig())
        result = service.analyze_run("run_abc123")
        service.export_results(result, "outputs/timing_report.json")
    """

    def __init__(
        self, 
        memory: Memory, 
        config: Optional[TimingAnalysisConfig] = None,
    ):
        """
        Initialize service.
        
        Args:
            memory: Database memory interface
            config: Optional configuration (uses defaults if None)
        """
        self.memory = memory
        self.config = config or TimingAnalysisConfig()
        self.cfg = get_config()
        
        # Set logging level
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
        
        logger.info(
            f"Initialized InterventionTimingService with "
            f"tau_hard={self._get_tau_hard():.4f}, "
            f"histogram_max={self.config.histogram_max_delta}"
        )

    def _get_tau_hard(self) -> float:
        """Get tau_hard threshold (config override or fallback)."""
        if self.config.tau_hard_override is not None:
            return float(self.config.tau_hard_override)
        return float(self.cfg.tau_hard)

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def analyze_run(self, run_id: str) -> TimingAnalysisResult:
        """
        Analyze intervention timing relative to collapse events.
        
        Args:
            run_id: Database run identifier
            
        Returns:
            TimingAnalysisResult with timing statistics and verdict
        """
        start_time = None  # Could add timing if needed
        logger.info(f"Starting timing analysis: run={run_id}")
        
        try:
            # 1. Load and prepare data
            df = self._load_and_prepare_data(run_id)
            if df is None or df.empty:
                return self._empty_result(run_id, "no_valid_data")
            
            # 2. Identify collapses
            collapse_iters = self._identify_collapses(df)
            if not collapse_iters:
                logger.warning(f"No collapses detected in run {run_id} (energy never exceeded tau_hard)")
                return self._empty_result(run_id, "no_collapses")
            
            # 3. Identify interventions
            interventions = self._identify_interventions(df)
            if interventions.empty:
                logger.warning(f"No interventions detected in run {run_id}")
                return self._empty_result(run_id, "no_interventions")
            
            logger.debug(
                f"Found {len(collapse_iters)} collapsed problems, "
                f"{len(interventions)} interventions"
            )
            
            # 4. Compute deltas (intervention → collapse)
            deltas = self._compute_deltas(interventions, collapse_iters)
            if len(deltas) == 0:
                logger.warning(
                    f"No interventions occurred before collapse in run {run_id}. "
                    f"All interventions were after collapse or in non-collapsing problems."
                )
                return self._empty_result(run_id, "no_pre_collapse_interventions")
            
            logger.debug(f"Analyzed {len(deltas)} pre-collapse interventions")
            
            # 5. Compute statistics
            stats = self._compute_statistics(deltas)
            
            # 6. Build histogram
            histogram = self._build_histogram(deltas)
            
            # 7. Compute verdict
            verdict = self._compute_verdict(stats)
            
            # 8. Compile result
            result = TimingAnalysisResult(
                run_id=run_id,
                status="success",
                mean_delta=float(np.mean(deltas)),
                median_delta=float(np.median(deltas)),
                std_delta=float(np.std(deltas)),
                pct_delta_le_1=float(np.mean(deltas <= 1) * 100),
                pct_delta_le_2=float(np.mean(deltas <= 2) * 100),
                pct_delta_ge_3=float(np.mean(deltas >= 3) * 100),
                n_interventions=len(interventions),
                n_collapsed_problems=len(collapse_iters),
                n_analyzed_interventions=len(deltas),
                histogram=histogram,
                verdict=verdict,
                diagnostics={
                    "tau_hard_used": self._get_tau_hard(),
                    "collapse_iterations": collapse_iters,
                    "intervention_actions": interventions["policy_action"].value_counts().to_dict(),
                    "delta_min": int(np.min(deltas)),
                    "delta_max": int(np.max(deltas)),
                },
                config=self.config.to_dict(),
            )
            
            # 9. Export if enabled
            if self.config.export_results:
                export_path = self.config.export_path or f"{self.cfg.reports_dir}/timing_{run_id}.json"
                self.export_results(result, export_path)
                logger.info(f"Exported timing results to {export_path}")
            
            logger.info(
                f"Timing analysis complete: run={run_id}, "
                f"median_delta={result.median_delta:.2f}, "
                f"verdict={verdict}"
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Timing analysis failed for run {run_id}: {e}")
            return TimingAnalysisResult(
                run_id=run_id,
                status="error",
                mean_delta=None,
                median_delta=None,
                std_delta=None,
                pct_delta_le_1=None,
                pct_delta_le_2=None,
                pct_delta_ge_3=None,
                n_interventions=0,
                n_collapsed_problems=0,
                n_analyzed_interventions=0,
                histogram={},
                verdict=f"ERROR: {str(e)}",
                diagnostics={"error": str(e)},
                config=self.config.to_dict(),
            )

    def export_results(self, result: TimingAnalysisResult, output_path: str) -> Path:
        """
        Export analysis results to JSON file.
        
        Args:
            result: Analysis result from analyze_run()
            output_path: Path to write JSON file
            
        Returns:
            Path to written file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use safe JSON serialization
        json_str = dumps_safe(result.to_dict(), indent=2)
        path.write_text(json_str, encoding="utf-8")
        
        logger.debug(f"Exported results to {path}")
        return path

    # ─────────────────────────────────────────────────────────
    # Internal Methods
    # ─────────────────────────────────────────────────────────

    def _load_and_prepare_data(self, run_id: str) -> Optional[pd.DataFrame]:
        """
        Load steps and prepare DataFrame for analysis.
        
        Returns:
            DataFrame with required columns or None if loading failed
        """
        steps = self.memory.steps.get_by_run(run_id)
        if not steps:
            logger.warning(f"No steps found for run {run_id}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([s.model_dump() for s in steps])
        
        # Validate required columns
        required = {"problem_id", "iteration", "total_energy", "policy_action"}
        missing = required - set(df.columns)
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return None
        
        # Sort by problem and iteration for correct temporal ordering
        df = df.sort_values(["problem_id", "iteration"]).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} steps for run {run_id}")
        return df

    def _identify_collapses(self, df: pd.DataFrame) -> Dict[int, int]:
        """
        Identify first collapse iteration per problem.
        
        Returns:
            Dict[problem_id -> first_collapse_iteration]
        """
        tau_hard = self._get_tau_hard()
        
        # Mark collapse steps
        df["is_collapse"] = df["total_energy"] > tau_hard
        
        # Get first collapse iteration per problem
        collapse_mask = df["is_collapse"]
        if not collapse_mask.any():
            return {}
        
        collapse_iters = (
            df[collapse_mask]
            .groupby("problem_id")["iteration"]
            .min()
            .to_dict()
        )
        
        logger.debug(f"Identified {len(collapse_iters)} collapsed problems (tau_hard={tau_hard:.4f})")
        return collapse_iters

    def _identify_interventions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify all intervention steps (non-ACCEPT actions).
        
        Returns:
            DataFrame with intervention records
        """
        interventions = df[df["policy_action"] != "ACCEPT"][
            ["problem_id", "iteration", "policy_action"]
        ].copy()
        
        if not interventions.empty:
            logger.debug(
                f"Found {len(interventions)} interventions: "
                f"{interventions['policy_action'].value_counts().to_dict()}"
            )
        
        return interventions

    def _compute_deltas(
        self, 
        interventions: pd.DataFrame, 
        collapse_iters: Dict[int, int]
    ) -> np.ndarray:
        """
        Compute delta = collapse_iteration - intervention_iteration for each intervention.
        
        Only includes interventions that occurred BEFORE collapse in problems that collapsed.
        
        Returns:
            Array of delta values (positive = intervention before collapse)
        """
        deltas = []
        
        for _, row in interventions.iterrows():
            pid = int(row["problem_id"])
            iter_idx = int(row["iteration"])
            
            # Skip if problem never collapsed
            if pid not in collapse_iters:
                continue
            
            collapse_iter = int(collapse_iters[pid])
            
            # Skip if intervention happened at or after collapse
            if iter_idx >= collapse_iter:
                continue
            
            # Compute delta (positive = intervention before collapse)
            delta = collapse_iter - iter_idx
            deltas.append(delta)
        
        return np.array(deltas, dtype=int) if deltas else np.array([], dtype=int)

    def _compute_statistics(self, deltas: np.ndarray) -> Dict[str, float]:
        """Compute summary statistics for delta distribution."""
        return {
            "mean": float(np.mean(deltas)),
            "median": float(np.median(deltas)),
            "std": float(np.std(deltas)),
            "min": int(np.min(deltas)),
            "max": int(np.max(deltas)),
            "pct_le_1": float(np.mean(deltas <= 1) * 100),
            "pct_le_2": float(np.mean(deltas <= 2) * 100),
            "pct_ge_3": float(np.mean(deltas >= 3) * 100),
        }

    def _build_histogram(self, deltas: np.ndarray) -> Dict[int, int]:
        """Build histogram of delta values."""
        max_delta = min(self.config.histogram_max_delta, int(np.max(deltas)) + 1)
        bins = range(0, max_delta + 2)  # Include upper edge
        
        counts, bin_edges = np.histogram(deltas, bins=bins)
        
        # Convert to dict: delta_value -> count
        histogram = {int(bin_edges[i]): int(counts[i]) for i in range(len(counts))}
        
        return histogram

    def _compute_verdict(self, stats: Dict[str, float]) -> str:
        """
        Compute human-readable verdict based on timing statistics.
        
        Verdicts:
        - TOO_LATE: >50% of interventions within 1 step of collapse
        - EARLY_ENOUGH: >40% of interventions with ≥3 steps margin
        - MIXED: Median ≥2 steps but doesn't meet other criteria
        - SUBOPTIMAL: Default case
        """
        pct_le_1 = stats.get("pct_le_1", 0)
        pct_ge_3 = stats.get("pct_ge_3", 0)
        median = stats.get("median", 0)
        
        if pct_le_1 > self.config.too_late_pct:
            return (
                f"⚠️  INTERVENING TOO LATE "
                f"(>{self.config.too_late_pct}% within {self.config.too_late_threshold} step of collapse)"
            )
        elif pct_ge_3 > self.config.early_enough_pct:
            return (
                f"✅ INTERVENING EARLY ENOUGH "
                f"(>{self.config.early_enough_pct}% with ≥{self.config.early_enough_threshold} steps margin)"
            )
        elif median >= self.config.early_enough_threshold - 1:
            return f"⚠️  TIMING MIXED (median {median:.1f} steps)"
        else:
            return f"⚠️  TIMING SUBOPTIMAL (median {median:.1f} < {self.config.early_enough_threshold - 1} steps)"

    def _empty_result(self, run_id: str, reason: str) -> TimingAnalysisResult:
        """Return standardized empty result for skipped analyses."""
        logger.info(f"⏭️ Skipping timing analysis for {run_id}: {reason}")
        
        return TimingAnalysisResult(
            run_id=run_id,
            status="skipped",
            mean_delta=None,
            median_delta=None,
            std_delta=None,
            pct_delta_le_1=None,
            pct_delta_le_2=None,
            pct_delta_ge_3=None,
            n_interventions=0,
            n_collapsed_problems=0,
            n_analyzed_interventions=0,
            histogram={},
            verdict=f"SKIPPED: {reason}",
            diagnostics={"skip_reason": reason},
            config=self.config.to_dict(),
        )


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Intervention Timing Analysis Service")
    parser.add_argument("--run-id", type=str, help="Run ID to analyze")
    parser.add_argument("--recent", action="store_true", help="Analyze most recent run")
    parser.add_argument("--tau-hard", type=float, help="Override tau_hard threshold")
    parser.add_argument("--export", action="store_true", help="Export results to JSON")
    parser.add_argument("--output-path", type=str, help="Output path for exported results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    cfg = get_config()
    memory = Memory(cfg.db_url)
    
    # Determine run ID
    if args.run_id:
        run_id = args.run_id
    else:
        runs = memory.runs.get_recent_runs(limit=1)
        if not runs:
            print("❌ No runs found")
            return
        run_id = runs[0].run_id
    
    print(f"Analyzing run: {run_id}")
    
    # Initialize service with CLI options
    config = TimingAnalysisConfig(
        tau_hard_override=args.tau_hard,
        export_results=args.export,
        export_path=args.output_path,
    )
    
    service = InterventionTimingService(memory, config=config)
    result = service.analyze_run(run_id)
    
    # Print summary
    print("\n" + "=" * 70)
    print("INTERVENTION TIMING ANALYSIS")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Status: {result.status.upper()}")
    print(f"Verdict: {result.verdict}")
    
    if result.status == "success":
        print(f"\n📊 Sample sizes:")
        print(f"  Total interventions: {result.n_interventions}")
        print(f"  Collapsed problems: {result.n_collapsed_problems}")
        print(f"  Analyzed (pre-collapse): {result.n_analyzed_interventions}")
        
        print(f"\n⏱️  Timing statistics:")
        print(f"  Mean delta: {result.mean_delta:.2f} steps")
        print(f"  Median delta: {result.median_delta:.2f} steps")
        print(f"  Std delta: {result.std_delta:.2f} steps")
        print(f"  Range: [{result.diagnostics['delta_min']}, {result.diagnostics['delta_max']}]")
        
        print(f"\n📈 Distribution:")
        print(f"  ≤{service.config.too_late_threshold} step to collapse: {result.pct_delta_le_1:.1f}%")
        print(f"  ≤2 steps to collapse: {result.pct_delta_le_2:.1f}%")
        print(f"  ≥{service.config.early_enough_threshold} steps to collapse: {result.pct_delta_ge_3:.1f}%")
        
        # Print histogram
        if result.histogram:
            print(f"\n📊 Histogram (delta → count):")
            for delta in sorted(result.histogram.keys()):
                count = result.histogram[delta]
                bar = "█" * min(count, 50)  # Scale for display
                print(f"  {delta:2d}: {bar} ({count})")
    
    elif result.status == "skipped":
        print(f"\n⚠️  Analysis skipped: {result.verdict}")
        print(f"    Diagnostics: {result.diagnostics}")
    
    elif result.status == "error":
        print(f"\n❌ Analysis failed: {result.verdict}")
    
    print("=" * 70)
    
    # Export info
    if args.export and result.status == "success":
        print(f"\n✅ Results exported (see configured output path)")


if __name__ == "__main__":
    main()