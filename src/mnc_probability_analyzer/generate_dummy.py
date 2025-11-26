import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_dummy(output_path: str, n: int = 200, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    # Create plausible ranges
    cgpa = rng.uniform(5.0, 10.0, n)                     # 5-10
    tenth = rng.uniform(60, 100, n)                      # 60-100
    twelfth = rng.uniform(60, 100, n)
    leetcode = rng.integers(0, 400, n)
    total_problems = leetcode + rng.integers(50, 500, n)
    tech_projects = rng.integers(0, 8, n)
    internships = rng.integers(0, 4, n)
    certs = rng.integers(0, 6, n)
    total_skills = rng.integers(3, 20, n)
    public_speaking = rng.integers(1, 5, n)
    teamwork_exp = rng.integers(1, 5, n)
    comm_skills = rng.integers(1, 5, n)

    df = pd.DataFrame({
        'CGPA': cgpa,
        '10th %': tenth,
        '12th %': twelfth,
        'LeetCode Solved': leetcode,
        'Total Problems Solved': total_problems,
        'Technical Projects': tech_projects,
        'Internships': internships,
        'Certifications': certs,
        'Total Skills': total_skills,
        'Public Speaking': public_speaking,
        'Teamwork Experience': teamwork_exp,
        'Comm Skills (1-5)': comm_skills,
    })

    # Save as a processed-like file that train.py expects
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dummy processed dataset for training and demos.')
    parser.add_argument('--output', default='data/processed/final_dataset.xlsx', help='Output .xlsx path')
    parser.add_argument('--rows', type=int, default=200, help='Number of rows to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    generate_dummy(args.output, args.rows, args.seed)
