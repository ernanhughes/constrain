# debug_features.py
import pandas as pd
from constrain.data.memory import Memory
from constrain.config import get_config

def debug_feature_extraction(limit=1000):
    memory = Memory(get_config().db_url)
    
    # Fetch steps like training does
    df = memory.steps.get_recent_unique_steps(limit=limit, exclude_policy_ids=[99])
    print(f"Steps fetched: {len(df)}")
    print(f"Columns in steps: {list(df.columns)}")
    
    # Try to fetch problem_summaries
    run_ids = df["run_id"].unique().tolist()
    try:
        summaries = memory.problem_summaries.get_by_run_ids(run_ids)
        if summaries:
            summaries_df = pd.DataFrame([{
                "run_id": s.run_id,
                "problem_id": s.problem_id,
                "intervention_helped": s.intervention_helped,
                "final_correct": s.final_correct,
            } for s in summaries])
            print(f"Problem summaries fetched: {len(summaries_df)}")
            
            # Merge
            df = df.merge(summaries_df, on=["run_id", "problem_id"], how="left")
            print(f"Columns after merge: {len(df.columns)}")
        else:
            print("⚠️  No problem_summaries found")
    except Exception as e:
        print(f"⚠️  problem_summaries query failed: {e}")
    
    # Check numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    print(f"\nNumeric columns ({len(numeric_cols)}):")
    for c in numeric_cols:
        non_null = df[c].notna().sum()
        print(f"  {c}: {non_null}/{len(df)} non-null")
    
    # Simulate feature filtering
    exclude = {
        "run_id", "problem_id", "iteration", "timestamp",
        "reasoning_text", "gold_answer", "extracted_answer",
        "prompt_text", "phase", "policy_action",
        "intervention_helped", "final_correct", "id",
    }
    feature_cols = [c for c in df.columns if c not in exclude]
    print(f"\nFeature columns after filtering: {len(feature_cols)}")
    print(feature_cols)
    
    return df

if __name__ == "__main__":
    debug_feature_extraction()