import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


def preprocess(input_path: str, output_path: str) -> None:
    df = pd.read_excel(input_path)

    # Drop direct identifiers / irrelevant columns if present
    for col in ['Email ID', 'SAP ID', 'Timestamp']:
        if col in df.columns:
            df = df.drop(col, axis=1, errors='ignore')

    # Combine skill columns into a count
    tech_col = 'Which technical skills do you have?'
    other_col = 'Other skills: '
    if tech_col in df.columns and other_col in df.columns:
        def count_skills(series):
            return series.fillna('').astype(str).apply(
                lambda x: len([s.strip() for s in x.split(',') if s.strip()])
            )
        df['Total Skills'] = count_skills(df[tech_col]) + count_skills(df[other_col])
        df = df.drop([tech_col, other_col], axis=1)

    # Remove any header-like duplicate first row if present (optional heuristic)
    if df.shape[0] > 0 and isinstance(df.iloc[0]['Full Name'] if 'Full Name' in df.columns else None, str) and df.iloc[0]['Full Name'] == 'Full Name':
        df = df.iloc[1:].reset_index(drop=True)

    # Clean specific columns
    for col in ['Technical Projects', 'Internships']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Example fixes for known records (optional, safe-guarded)
    if 'Full Name' in df.columns:
        fixes = {
            ('Arnav Sharma', 'LeetCode Solved'): 25,
            ('Siya Chauhan', 'LeetCode Solved'): 25,
            ('Siya Chauhan', 'Total Problems Solved'): 400,
        }
        for (name, col), val in fixes.items():
            if col in df.columns:
                df.loc[df['Full Name'] == name, col] = val

    # Normalize selected numeric columns to [0,1]
    numeric_cols = [c for c in ['CGPA', '10th %', '12th %'] if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if numeric_cols:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Additional normalization for modeling convenience
    cols_to_normalize = [c for c in ['Technical Projects', 'Internships', 'Total Problems Solved', 'CGPA', '10th %', '12th %'] if c in df.columns]
    if cols_to_normalize:
        scaler2 = MinMaxScaler()
        df[cols_to_normalize] = scaler2.fit_transform(df[cols_to_normalize])

    # Compute readiness proxy features if source columns exist
    def safe(df, col):
        return df[col] if col in df.columns else 0

    if {'LeetCode Solved','Technical Projects','Total Problems Solved','CGPA'} <= set(df.columns):
        df['Google_Readiness'] = (
            0.4 * df['LeetCode Solved'] +
            0.3 * df['Technical Projects'] +
            0.2 * df['Total Problems Solved'] +
            0.1 * df['CGPA']
        ).clip(0, 1)

    if {'LeetCode Solved','Technical Projects','Internships','CGPA'} <= set(df.columns):
        df['Amazon_Readiness'] = (
            0.35 * df['LeetCode Solved'] +
            0.25 * df['Technical Projects'] +
            0.25 * df['Internships'] +
            0.15 * df['CGPA']
        ).clip(0, 1)

    if {'CGPA','10th %','12th %','Technical Projects'} <= set(df.columns):
        df['Infosys_Readiness'] = (
            0.3 * df['CGPA'] + 0.2 * df['10th %'] + 0.2 * df['12th %'] + 0.3 * df['Technical Projects']
        ).clip(0, 1)

    if {'Technical Projects','LeetCode Solved','Total Problems Solved','CGPA'} <= set(df.columns):
        df['Microsoft_Readiness'] = (
            0.4 * df['Technical Projects'] + 0.3 * df['LeetCode Solved'] + 0.2 * df['Total Problems Solved'] + 0.1 * df['CGPA']
        ).clip(0, 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess survey data into a modeling dataset.')
    parser.add_argument('--input', required=True, help='Path to raw .xlsx file')
    parser.add_argument('--output', required=True, help='Path to write processed .xlsx file')
    args = parser.parse_args()

    preprocess(args.input, args.output)
