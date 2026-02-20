import json

from constrain.config import get_config
from constrain.data.memory import Memory
from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator
from constrain.analysis.stage3.engine.signal_modeling_engine import SignalModelingEngine
from constrain.analysis.stage3.engine.shap_explainer import ShapExplainer


RUN_ID = "run_18cdc06e"  # change if needed


def main():

    print("🔎 Initializing memory...")
    config = get_config()
    memory = Memory(config.db_url)

    print("📊 Building dataframe...")
    df = MetricsAggregator.build_run_dataframe(memory, RUN_ID)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # Choose target (adjust if needed)
    TARGET = "correctness"

    print("Columns:")
    print(df.columns.tolist())

    engine = SignalModelingEngine(n_splits=5)

    print("🚀 Running signal modeling engine...")
    results = engine.run(
        df=df,
        target_col=TARGET,
        group_col="problem_id",
        exclude_cols=[
            "step_id",
            "run_id",
            "problem_id",
            "iteration",
            "policy_action",
            "phase",
            "accuracy",
            "extracted_answer",
        ],
    )

    print("\n📈 Engine Results:")
    print(json.dumps(results, indent=2))

    # Optional SHAP test on full model
    print("\n🔬 Running SHAP on final model...")

    # Fit final model on full data
    X, y, _ = engine._prepare_features(
        df,
        TARGET,
        "problem_id",
        exclude_cols=[
            "step_id",
            "run_id",
            "problem_id",
            "iteration",
            "policy_action",
            "phase",
            "accuracy",
            "extracted_answer",
        ],
    )

    final_model = engine._model_factory()
    final_model.fit(X, y)

    shap_ranking = ShapExplainer.explain(final_model, X)

    print("\n🔥 Top 10 SHAP Features:")
    for f, v in shap_ranking[:10]:
        print(f"{f}: {v:.6f}")


if __name__ == "__main__":
    main()