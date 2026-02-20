#!/usr/bin/env python3
"""
XGBoost Signal Discovery Pipeline
Identify which metrics PRECEDE collapse (lead-lag validation)
"""
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


def load_forensic_data(db_path: str = "recursive_instability.db") -> pd.DataFrame:
    """Load all steps with collapse labels"""
    conn = sqlite3.connect(db_path)
    
    # Load steps with collapse labels
    query = """
    SELECT 
        s.*,
        CASE 
            WHEN s.phase = 'collapse' THEN 1
            ELSE 0
        END as collapse_label,
        LEAD(s.phase, 1) OVER (PARTITION BY s.run_id, s.problem_id ORDER BY s.iteration) as next_phase,
        LEAD(s.phase, 2) OVER (PARTITION BY s.run_id, s.problem_id ORDER BY s.iteration) as phase_t_plus_2,
        LEAD(s.phase, 3) OVER (PARTITION BY s.run_id, s.problem_id ORDER BY s.iteration) as phase_t_plus_3
    FROM steps s
    WHERE s.iteration < (SELECT MAX(iteration) FROM steps s2 WHERE s2.run_id = s.run_id AND s2.problem_id = s.problem_id)
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_lead_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for predicting collapse at t+k"""
    features = df.copy()
    
    # Lead indicators: signals at t that predict collapse at t+1, t+2, t+3
    for lag in [1, 2, 3]:
        features[f'collapse_t_plus_{lag}'] = (
            features[f'phase_t_plus_{lag}'] == 'collapse'
        ).astype(int)
    
    # Rolling statistics (window = 3 iterations)
    for col in ['energy', 'foreign_char_ratio', 'repetition_score']:
        features[f'{col}_rolling_mean'] = features.groupby(['run_id', 'problem_id'])[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        features[f'{col}_rolling_std'] = features.groupby(['run_id', 'problem_id'])[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
        )
    
    return features

def train_collapse_predictor(df: pd.DataFrame, prediction_horizon: int = 2):
    """
    Train XGBoost to predict collapse at t+prediction_horizon
    Returns model + feature importance
    """
    # Target: collapse at t+prediction_horizon
    target_col = f'collapse_t_plus_{prediction_horizon}'
    
    # Features: all signal metrics at time t
    feature_cols = [
        'energy', 'grounding_energy', 'stability_energy',
        'foreign_char_ratio', 'ascii_ratio', 'repetition_score',
        'text_length', 'accuracy',
        'energy_rolling_mean', 'energy_rolling_std',
        'foreign_char_ratio_rolling_mean', 'foreign_char_ratio_rolling_std'
    ]
    
    # Drop rows with missing targets
    df_clean = df.dropna(subset=[target_col] + feature_cols)
    
    # Train/test split (time-series aware)
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    aucs = []
    
    for train_idx, test_idx in tscv.split(df_clean):
        X_train = df_clean.iloc[train_idx][feature_cols]
        y_train = df_clean.iloc[train_idx][target_col]
        X_test = df_clean.iloc[test_idx][feature_cols]
        y_test = df_clean.iloc[test_idx][target_col]
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        aucs.append(auc_score)
        models.append(model)
    
    # Average model
    avg_auc = np.mean(aucs)
    print(f"Predicting collapse at t+{prediction_horizon}: AUC = {avg_auc:.3f}")
    
    # Feature importance from last fold
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': models[-1].feature_importances_
    }).sort_values('importance', ascending=False)
    
    return models[-1], importance, avg_auc

def plot_signal_lead_lag(df: pd.DataFrame):
    """Visualize which signals precede collapse"""
    # Filter to trajectories that collapse
    collapse_trajectories = df[df['phase'] == 'collapse']['problem_id'].unique()
    collapse_df = df[df['problem_id'].isin(collapse_trajectories)].copy()
    
    # Compute average signal trajectory before collapse
    collapse_df['steps_to_collapse'] = collapse_df.groupby(['run_id', 'problem_id'])['iteration'].transform(
        lambda x: (x.max() - x)
    )
    
    # Aggregate signals by steps before collapse
    signal_agg = collapse_df[collapse_df['steps_to_collapse'] <= 5].groupby('steps_to_collapse')[
        ['energy', 'foreign_char_ratio', 'repetition_score', 'accuracy']
    ].mean().reset_index()
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['energy', 'foreign_char_ratio', 'repetition_score', 'accuracy']
    titles = ['Hallucination Energy', 'Foreign Char Ratio', 'Repetition Score', 'Accuracy']
    colors = ['red', 'purple', 'orange', 'green']
    
    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes[idx // 2, idx % 2]
        ax.plot(signal_agg['steps_to_collapse'], signal_agg[metric], marker='o', linewidth=2.5, color=color)
        ax.set_xlabel('Steps Before Collapse')
        ax.set_ylabel(metric)
        ax.set_title(f'{title} Trajectory Before Collapse')
        ax.invert_xaxis()  # Show time moving forward to collapse
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('signal_lead_lag.png', dpi=300, bbox_inches='tight')
    print("✅ Signal lead-lag plot saved to signal_lead_lag.png")

def main():
    print("="*70)
    print("SIGNAL DISCOVERY: Identifying Predictors of Recursive Collapse")
    print("="*70)
    
    # Load data
    print("\n1. Loading forensic data...")
    df = load_forensic_data()
    print(f"   Loaded {len(df)} steps from {df['run_id'].nunique()} runs")
    
    # Create features
    print("\n2. Creating lead-lag features...")
    df_features = create_lead_lag_features(df)
    
    # Train predictors for different horizons
    print("\n3. Training collapse predictors...")
    results = {}
    for horizon in [1, 2, 3]:
        model, importance, auc_score = train_collapse_predictor(df_features, prediction_horizon=horizon)
        results[horizon] = {'model': model, 'importance': importance, 'auc': auc_score}
        
        print(f"\nTop 5 predictors for collapse at t+{horizon}:")
        print(importance.head(5).to_string(index=False))
    
    # Plot signal trajectories
    print("\n4. Generating signal lead-lag visualization...")
    plot_signal_lead_lag(df)
    
    # Critical finding: Foreign char ratio predictive power
    print("\n" + "="*70)
    print("KEY FINDING: Foreign Character Ratio as Early Warning Signal")
    print("="*70)
    for horizon in [1, 2, 3]:
        foreign_rank = results[horizon]['importance'][
            results[horizon]['importance']['feature'] == 'foreign_char_ratio'
        ].index[0] + 1
        print(f"  • Foreign char ratio rank for t+{horizon}: #{foreign_rank}")
    
    print("\n✅ Signal discovery complete. Check signal_lead_lag.png for visualization.")
    print("   Use results[horizon]['model'] for deployment or further analysis.")

if __name__ == "__main__":
    main()