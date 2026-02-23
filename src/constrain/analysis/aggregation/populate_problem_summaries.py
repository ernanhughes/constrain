# constrain/analysis/aggregation/populate_problem_summaries.py
"""
Populates problem_summaries table from steps + interventions data.
Run this after each simulation completes, or as a backfill for existing runs.
"""

from sqlalchemy import func
from constrain.data.memory import Memory
from constrain.data.orm.problem_summary import ProblemSummaryORM
from constrain.data.orm.step import StepORM
from constrain.data.orm.intervention import InterventionORM
from constrain.config import get_config


def compute_problem_summary(memory: Memory, run_id: str, problem_id: int) -> dict:
    """
    Compute problem-level summary from steps and interventions.
    """
    def op(s):
        # Get all steps for this problem
        steps = (
            s.query(StepORM)
            .filter(
                StepORM.run_id == run_id,
                StepORM.problem_id == problem_id,
            )
            .order_by(StepORM.iteration)
            .all()
        )

        if not steps:
            return None

        # Get all interventions for this problem
        interventions = (
            s.query(InterventionORM)
            .filter(
                InterventionORM.run_id == run_id,
                InterventionORM.problem_id == problem_id,
            )
            .all()
        )

        # Compute metrics
        final_accuracy = steps[-1].accuracy if steps[-1].accuracy is not None else 0.0
        final_correct = int(final_accuracy > 0.5)

        initial_accuracy = steps[0].accuracy if steps[0].accuracy is not None else 0.0

        num_interventions = len(interventions)
        any_intervention = num_interventions > 0

        avg_energy = sum(s.total_energy for s in steps) / len(steps)
        max_energy = max(s.total_energy for s in steps)
        num_iterations = len(steps)

        # ─────────────────────────────────────────────────────────────
        # Did intervention help? (CRITICAL for utility head training)
        # ─────────────────────────────────────────────────────────────
        # Compare accuracy before vs after intervention
        intervention_helped = False
        intervention_harmed = False

        if any_intervention:
            # Find first intervention iteration
            first_intervention_iter = min(i.iteration for i in interventions)

            # Accuracy before intervention
            pre_intervention_steps = [
                s for s in steps if s.iteration < first_intervention_iter
            ]
            pre_accuracy = (
                sum(s.accuracy for s in pre_intervention_steps) / len(pre_intervention_steps)
                if pre_intervention_steps else 0.0
            )

            # Accuracy after intervention
            post_intervention_steps = [
                s for s in steps if s.iteration >= first_intervention_iter
            ]
            post_accuracy = (
                sum(s.accuracy for s in post_intervention_steps) / len(post_intervention_steps)
                if post_intervention_steps else 0.0
            )

            # Did accuracy improve after intervention?
            if post_accuracy > pre_accuracy + 0.05:  # 5% threshold
                intervention_helped = True
            elif post_accuracy < pre_accuracy - 0.05:
                intervention_harmed = True

        return {
            "run_id": run_id,
            "problem_id": problem_id,
            "final_correct": bool(final_correct),
            "any_intervention": any_intervention,
            "num_interventions": num_interventions,
            "avg_energy": float(avg_energy),
            "max_energy": float(max_energy),
            "num_iterations": num_iterations,
            "intervention_helped": bool(intervention_helped),
            "intervention_harmed": bool(intervention_harmed),
        }

    return memory.steps._run(op)


def populate_for_run(memory: Memory, run_id: str) -> int:
    """
    Populate problem_summaries for all problems in a run.
    Returns count of summaries created.
    """
    def op(s):
        # Get all unique problem_ids for this run
        problem_ids = (
            s.query(func.distinct(StepORM.problem_id))
            .filter(StepORM.run_id == run_id)
            .all()
        )
        problem_ids = [p[0] for p in problem_ids]

        count = 0
        for problem_id in problem_ids:
            # Check if summary already exists
            existing = (
                s.query(ProblemSummaryORM)
                .filter(
                    ProblemSummaryORM.run_id == run_id,
                    ProblemSummaryORM.problem_id == problem_id,
                )
                .first()
            )

            if existing:
                continue  # Skip if already populated

            # Compute summary
            summary_data = compute_problem_summary(memory, run_id, problem_id)

            if summary_data:
                # Insert into DB
                summary_orm = ProblemSummaryORM(**summary_data)
                s.add(summary_orm)
                count += 1

        s.commit()
        return count

    return memory.steps._run(op)


def backfill_all_runs(memory: Memory, limit: int = 100) -> int:
    """
    Backfill problem_summaries for recent runs.
    """
    def op(s):
        # Get recent run_ids
        run_ids = (
            s.query(func.distinct(StepORM.run_id))
            .order_by(StepORM.timestamp.desc())
            .limit(limit)
            .all()
        )
        run_ids = [r[0] for r in run_ids]

        total_count = 0
        for run_id in run_ids:
            count = populate_for_run(memory, run_id)
            total_count += count
            print(f"  Run {run_id}: {count} problem summaries created")

        return total_count

    return memory.steps._run(op)


def main():
    cfg = get_config()
    memory = Memory(cfg.db_url)

    print("="*60)
    print("POPULATING problem_summaries TABLE")
    print("="*60)

    # Option 1: Backfill recent runs
    print("\n📊 Backfilling recent runs...")
    total = backfill_all_runs(memory, limit=100)
    print(f"\n✅ Created {total} problem summaries")

    # Option 2: Or populate for specific run
    # run_id = "your_run_id_here"
    # count = populate_for_run(memory, run_id)
    # print(f"Created {count} problem summaries for run {run_id}")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()