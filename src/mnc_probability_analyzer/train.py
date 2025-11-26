import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def train_and_save_models(data_path: str, models_dir: str) -> None:
    df = pd.read_excel(data_path)

    # Drop non-feature columns if present
    drop_columns = [c for c in ['Full Name', 'Branch', 'Year', 'Suggested_Improvements'] if c in df.columns]
    if drop_columns:
        df = df.drop(columns=drop_columns)

    target_defs = {
        'Google':   {'target': 'Google_Readiness_noisy',   'features': ['CGPA', 'Total Problems Solved', 'LeetCode Solved'],
                     'base': 'Google_Readiness',    'weights': {'CGPA': 0.33, 'Total Problems Solved': 0.30, 'LeetCode Solved': 0.21}, 'noise': 0.15},
        'Microsoft':{'target': 'Microsoft_Readiness_noisy','features': ['Technical Projects', 'Internships', 'Certifications', 'Total Problems Solved', 'Total Skills'],
                     'base': 'Microsoft_Readiness', 'weights': {'Technical Projects': 0.2, 'Internships': 0.098, 'Certifications': 0.14, 'Total Problems Solved': 0.059, 'Total Skills': 0.062}, 'noise': 0.12},
        'Amazon':   {'target': 'Amazon_Readiness_noisy',   'features': ['Technical Projects', 'Internships', 'Certifications', 'LeetCode Solved', 'Teamwork Experience'],
                     'base': 'Amazon_Readiness',    'weights': {'Technical Projects': 0.25, 'Internships': 0.15, 'Certifications': 0.18, 'LeetCode Solved': 0.12, 'Teamwork Experience': 0.10}, 'noise': 0.10},
        'Infosys':  {'target': 'Infosys_Readiness_noisy',  'features': ['Internships', 'Technical Projects', 'Certifications', 'LeetCode Solved', 'Teamwork Experience'],
                     'base': 'Infosys_Readiness',   'weights': {'Internships': 0.28, 'Technical Projects': 0.20, 'Certifications': 0.14, 'LeetCode Solved': 0.15, 'Teamwork Experience': 0.10}, 'noise': 0.09},
    }

    # Recompute base readiness (scaled 0-1) then add noise to simulate label
    import numpy as np
    for name, spec in target_defs.items():
        base = spec['base']
        feats = spec['features']
        weights = spec['weights']
        if all(f in df.columns for f in weights.keys()):
            df[base] = sum(df[f] * w for f, w in weights.items())
            maxv = df[base].max()
            if maxv and maxv > 0:
                df[base] /= maxv
            noise = np.random.normal(0, spec['noise'], len(df))
            df[spec['target']] = (df[base] + noise).clip(0, 1)

    Path(models_dir).mkdir(parents=True, exist_ok=True)

    for name, spec in target_defs.items():
        feats = [f for f in spec['features'] if f in df.columns]
        target = spec['target']
        if not feats or target not in df.columns:
            continue
        X = df[feats]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name}: R2={r2_score(y_test, y_pred):.2f}  MAE={mean_absolute_error(y_test, y_pred):.2f}  MSE={mean_squared_error(y_test, y_pred):.2f}")
        joblib.dump({'model': model, 'features': feats}, str(Path(models_dir) / f"{name.lower()}.joblib"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train readiness models and save them to disk.')
    parser.add_argument('--data', required=True, help='Path to processed dataset (.xlsx)')
    parser.add_argument('--models_dir', default='models', help='Directory to save models')
    args = parser.parse_args()

    train_and_save_models(args.data, args.models_dir)
