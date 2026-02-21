import joblib
import pandas as pd
import xgboost as xgb

from constrain.data.memory import Memory
from constrain.analysis.aggregation.metrics_aggregator import MetricsAggregator
from constrain.config import get_config


def build_training_dataframe(run_id: str):
    memory = Memory(get_config().db_url)
    df = MetricsAggregator.build_run_dataframe(memory, run_id)

    cfg = get_config()
    tau_hard = cfg.tau_hard

    df = df.sort_values(["problem_id", "iteration"])

    rows = []

    for pid, group in df.groupby("problem_id"):
        group = group.reset_index(drop=True)

        for i in range(len(group) - 1):
            current = group.iloc[i]
            next_row = group.iloc[i + 1]

            collapse_next = int(next_row["total_energy"] > tau_hard)

            feature_cols = [
                c
                for c in df.columns
                if c
                not in [
                    "problem_id",
                    "iteration",
                    "reasoning_text",
                    "gold_answer",
                    "extracted_answer",
                ]
            ]

            row = current[feature_cols].to_dict()
            row["target"] = collapse_next

            rows.append(row)

    train_df = pd.DataFrame(rows)
    
    print(train_df["target"].value_counts(normalize=True))
    print(train_df["target"].value_counts(normalize=True))

    return train_df


def train_model(train_df, output_path):
    numeric_df = train_df.select_dtypes(include=["number"])
    y = numeric_df["target"]
    X = numeric_df.drop(columns=["target"])

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X, y)

    print("Training complete.")
    print("Feature count:", X.shape[1])

    print(f"model.predict_proba(X).mean(): {model.predict_proba(X).mean()}")
    print(f"model.predict_proba(X)[:,1].std(): {model.predict_proba(X)[:,1].std()}")

    joblib.dump(model, output_path)

    print("Model saved to:", output_path)


def main():
    run_id = "run_e9b4c143" 

    cfg = get_config()
    output_path = cfg.learned_model_path

    df = build_training_dataframe(run_id)

    print("Training samples:", len(df))
    print("Positive rate:", df["target"].mean())

    train_model(df, output_path)


if __name__ == "__main__":
    main()
